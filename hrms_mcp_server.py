import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
import re
from dotenv import load_dotenv
import httpx
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hrms-mcp-server")

# Configuration
HRMS_API_BASE_URL = "https://portalapi.kryptosinfosys.com"
# Get auth token from environment variable (set by main.py)
AUTH_TOKEN = os.getenv("HRMS_AUTH_TOKEN", "")

if not AUTH_TOKEN:
    logger.warning("No auth token provided via environment variable HRMS_AUTH_TOKEN")

class JWTHelper:
    """Helper class to decode JWT tokens"""
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """
        Decode JWT token without verification (for extracting empId)
        Note: In production, you should verify the token signature
        """
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            logger.info(f"Decoded JWT payload: {payload}")
            return payload
        except Exception as e:
            logger.error(f"Failed to decode JWT: {e}")
            return {}
    
    @staticmethod
    def get_emp_id_from_token(token: str) -> Optional[int]:
        """Extract empId from JWT token"""
        decoded = JWTHelper.decode_token(token)
        if decoded:
            # Try different common field names for employee ID
            # NOTE: 'uid' is checked first as it's the field used in this HRMS system
            for key in ['uid', 'empId', 'emp_id', 'employeeId', 'employee_id', 'sub', 'userId', 'user_id', 'id', 'user', 'EmployeeId']:
                if key in decoded:
                    try:
                        emp_id = int(decoded[key])
                        logger.info(f"Successfully extracted empId={emp_id} from JWT token field '{key}'")
                        return emp_id
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Found key '{key}' but couldn't convert to int: {decoded[key]}, error: {e}")
                        continue
            logger.error(f"Could not find empId in token. Available keys: {list(decoded.keys())}")
            logger.error(f"Token payload: {decoded}")
        return None

class DateParser:
    """Helper class to parse natural language dates"""
    
    @staticmethod
    def parse_date_from_text(text: str, reference_date: Optional[datetime] = None) -> Optional[str]:
        """
        Parse natural language date expressions and return ISO format date string
        
        Args:
            text: Natural language date text (e.g., "today", "yesterday", "tomorrow", "2025-12-28")
            reference_date: Reference date for relative calculations (defaults to today)
        
        Returns:
            ISO format date string (YYYY-MM-DD) or None if parsing fails
        """
        if not text:
            return None
            
        text = text.lower().strip()
        
        # Use current date as reference if not provided
        if reference_date is None:
            reference_date = datetime.now()
        
        # Check for explicit date format (YYYY-MM-DD)
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        date_match = re.search(date_pattern, text)
        if date_match:
            return date_match.group(0)
        
        # Handle relative dates
        if text == "today":
            return reference_date.strftime("%Y-%m-%d")
        elif text == "yesterday":
            return (reference_date - timedelta(days=1)).strftime("%Y-%m-%d")
        elif text == "tomorrow":
            return (reference_date + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "last week" in text:
            return (reference_date - timedelta(weeks=1)).strftime("%Y-%m-%d")
        elif "next week" in text:
            return (reference_date + timedelta(weeks=1)).strftime("%Y-%m-%d")
        elif "days ago" in text:
            # Extract number of days
            match = re.search(r'(\d+)\s*days?\s+ago', text)
            if match:
                days = int(match.group(1))
                return (reference_date - timedelta(days=days)).strftime("%Y-%m-%d")
        elif "days from now" in text or "days later" in text:
            # Extract number of days
            match = re.search(r'(\d+)\s*days?', text)
            if match:
                days = int(match.group(1))
                return (reference_date + timedelta(days=days)).strftime("%Y-%m-%d")
        
        # If no pattern matched, return None
        return None
    
    @staticmethod
    def extract_dates_from_prompt(prompt: str) -> Dict[str, Optional[str]]:
        """
        Extract fromDate and toDate from user prompt
        
        Returns:
            Dict with 'from_date' and 'to_date' keys
        """
        prompt_lower = prompt.lower()
        result = {"from_date": None, "to_date": None}
        
        # Look for explicit date ranges
        # Pattern: "from YYYY-MM-DD to YYYY-MM-DD"
        range_pattern = r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        range_match = re.search(range_pattern, prompt_lower)
        if range_match:
            result["from_date"] = range_match.group(1)
            result["to_date"] = range_match.group(2)
            return result
        
        # Check for single date expressions
        if "today" in prompt_lower:
            today = datetime.now().strftime("%Y-%m-%d")
            result["from_date"] = today
            result["to_date"] = today
        elif "yesterday" in prompt_lower:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            result["from_date"] = yesterday
            result["to_date"] = yesterday
        elif "tomorrow" in prompt_lower:
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            result["from_date"] = tomorrow
            result["to_date"] = tomorrow
        elif "this week" in prompt_lower:
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            result["from_date"] = start_of_week.strftime("%Y-%m-%d")
            result["to_date"] = end_of_week.strftime("%Y-%m-%d")
        elif "last week" in prompt_lower:
            today = datetime.now()
            start_of_last_week = today - timedelta(days=today.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            result["from_date"] = start_of_last_week.strftime("%Y-%m-%d")
            result["to_date"] = end_of_last_week.strftime("%Y-%m-%d")
        else:
            # Try to extract any YYYY-MM-DD dates
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            dates = re.findall(date_pattern, prompt_lower)
            if dates:
                result["from_date"] = dates[0]
                if len(dates) > 1:
                    result["to_date"] = dates[1]
        
        return result

class HRMSAPIClient:
    """Client to interact with the HRMS APIs"""
    
    def __init__(self, base_url: str, auth_token: str = ""):
        self.base_url = base_url
        self.auth_token = auth_token
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"HRMSAPIClient initialized with auth_token: {'***' if auth_token else 'NOT_SET'}")
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
    
    async def get_leave_history(self, payload: Dict[str, Any]):
        """Get leave history using POST request with payload"""
        try:
            logger.info(f"Fetching leave history with payload: {payload}")
            response = await self.client.post(
                f"{self.base_url}/api/Leave/history",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching leave history: {e}")
            return {"error": str(e)}
    
    async def get_leave_statuses(self):
        """Get all possible leave statuses"""
        try:
            headers = self._get_headers()
            logger.info(f"Fetching leave statuses with auth token present: {bool(self.auth_token)}")
            logger.info(f"Auth token (masked): {'***' + self.auth_token[-10:] if self.auth_token and len(self.auth_token) > 10 else 'NOT_SET'}")
            
            response = await self.client.get(
                f"{self.base_url}/api/Leave/statuses",
                headers=headers
            )
            logger.info(f"Leave statuses response status: {response.status_code}")
            
            if response.status_code == 401:
                logger.error("401 Unauthorized - Auth token may be invalid or expired")
                return {
                    "error": "Unauthorized - Authentication token is invalid or expired",
                    "status_code": 401,
                    "auth_token_present": bool(self.auth_token)
                }
            
            response.raise_for_status()
            result = response.json()
            logger.info(f"Successfully fetched {len(result) if isinstance(result, list) else 'unknown'} leave statuses")
            return result
        except Exception as e:
            logger.error(f"Error fetching leave statuses: {e}")
            return {
                "error": str(e),
                "auth_token_present": bool(self.auth_token)
            }
    
    async def get_leave_types(self):
        """Get all leave types"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/LeaveTypes",
                headers=self._get_headers()
            )
            logger.info(f"Leave types response status: {response.status_code}")
            logger.info(f"Leave types response body: {response.text[:500]}")
            response.raise_for_status()
            result = response.json()
            logger.info(f"Leave types result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error fetching leave types: {e}")
            return {"error": str(e), "status_code": getattr(response, 'status_code', None), "response_text": getattr(response, 'text', None)}
    
    async def get_leave_policies(self):
        """Get all leave policies"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/LeavePolicies",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching leave policies: {e}")
            return {"error": str(e)}
    
    async def get_leave_day_parts(self):
        """Get all possible leave day parts"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/Leave/leavedaypart",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching leave day parts: {e}")
            return {"error": str(e)}
    
    async def get_assign_shift(
        self,
        assign_shift_id: Optional[int] = None,
        search_term: Optional[str] = None,
        employee_id: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        page_number: Optional[int] = None,
        page_size: Optional[int] = None
    ):
        """Get assigned shifts based on filters"""
        try:
            params = {}
            if assign_shift_id is not None:
                params["assignShiftId"] = assign_shift_id
            if search_term is not None:
                params["searchTerm"] = search_term
            if employee_id is not None:
                params["employeeId"] = employee_id
            if from_date is not None:
                params["fromDate"] = from_date
            if to_date is not None:
                params["toDate"] = to_date
            if page_number is not None:
                params["pageNumber"] = page_number
            if page_size is not None:
                params["pageSize"] = page_size
            
            response = await self.client.get(
                f"{self.base_url}/api/AssignShift",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching assigned shifts: {e}")
            return {"error": str(e)}
    
    async def get_attendance_policies(
        self,
        policy_id: Optional[int] = None,
        is_active: Optional[bool] = None,
        query: Optional[str] = None
    ):
        """Get attendance policies"""
        try:
            params = {}
            if policy_id is not None:
                params["id"] = policy_id
            if is_active is not None:
                params["isActive"] = is_active
            if query is not None:
                params["q"] = query
            
            response = await self.client.get(
                f"{self.base_url}/api/attendance/policies",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching attendance policies: {e}")
            return {"error": str(e)}
    
    async def get_holidays(
        self,
        holiday_type_filter: Optional[str] = None,
        year: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ):
        """Get holidays based on filters"""
        try:
            params = {}
            if holiday_type_filter is not None:
                params["HolidayTypeFilter"] = holiday_type_filter
            if year is not None:
                params["Year"] = year
            if from_date is not None:
                params["FromDate"] = from_date
            if to_date is not None:
                params["ToDate"] = to_date
            
            response = await self.client.get(
                f"{self.base_url}/api/Holidays",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching holidays: {e}")
            return {"error": str(e)}
    
    async def get_company(
        self,
        company_id: Optional[str] = None,
        page_no: Optional[int] = None,
        page_size: Optional[int] = None
    ):
        """Get company information"""
        try:
            params = {}
            if company_id is not None:
                params["id"] = company_id
            if page_no is not None:
                params["pageNo"] = page_no
            if page_size is not None:
                params["pageSize"] = page_size
            
            response = await self.client.get(
                f"{self.base_url}/api/Companies",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching company info: {e}")
            return {"error": str(e)}
    
    async def apply_leave(self, leave_data: Dict[str, Any]):
        """Apply for leave with the new API structure"""
        try:
            logger.info("="*60)
            logger.info("API CLIENT: Applying leave")
            logger.info(f"URL: {self.base_url}/api/Leave/apply")
            logger.info(f"Payload: {leave_data}")
            logger.info(f"Auth token present: {bool(self.auth_token)}")
            if self.auth_token:
                logger.info(f"Auth token (first 20 chars): {self.auth_token[:20]}...")
            logger.info("="*60)
            
            response = await self.client.post(
                f"{self.base_url}/api/Leave/apply",
                json=leave_data,
                headers=self._get_headers()
            )
            
            logger.info(f"API Response Status: {response.status_code}")
            logger.info(f"API Response Headers: {dict(response.headers)}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"✅ Leave application successful!")
            logger.info(f"API Response Body: {result}")
            
            return {"success": True, "data": result}
            
        except httpx.HTTPStatusError as e:
            logger.error("="*60)
            logger.error(f"❌ HTTP error applying leave")
            logger.error(f"Status Code: {e.response.status_code}")
            logger.error(f"Response Headers: {dict(e.response.headers)}")
            logger.error(f"Response Body: {e.response.text}")
            logger.error("="*60)
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "status_code": e.response.status_code
            }
        except Exception as e:
            logger.error("="*60)
            logger.error(f"❌ Exception applying leave: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error("="*60)
            return {"success": False, "error": str(e)}

# Initialize FastMCP server and API client
mcp = FastMCP("hrms-management")
api_client = HRMSAPIClient(HRMS_API_BASE_URL, AUTH_TOKEN)

# ==== LEAVE TOOLS ====

@mcp.tool()
async def get_leave_history(
    prompt: str = "",
    emp_id: Optional[int] = None,
    reporting_manager_id: Optional[int] = None,
    status: Optional[int] = None,
    page_number: int = 1,
    page_size: int = 10,
    sort_order: Optional[str] = None,
    sort_by: Optional[str] = None,
    search_term: Optional[str] = None,
    company_id: Optional[int] = None,
    branch_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get leave history based on filters. Supports natural language date queries.
    
    Parameters:
    - prompt: Natural language prompt (e.g., "who is on leave today", "show leaves for yesterday")
    - emp_id: Employee ID (null to get all employees' leaves)
    - reporting_manager_id: Filter by reporting manager
    - status: Leave status filter (null for all statuses)
    - page_number: Page number for pagination (default: 1)
    - page_size: Number of records per page (default: 10, can be adjusted based on prompt)
    - sort_order: Sort order (e.g., "asc" or "desc")
    - sort_by: Field to sort by
    - search_term: Search term to filter results
    - company_id: Company ID filter
    - branch_id: Branch ID filter
    - from_date: Start date filter (YYYY-MM-DD format, auto-calculated from prompt if not provided)
    - to_date: End date filter (YYYY-MM-DD format, auto-calculated from prompt if not provided)
    
    Examples:
    - "Who is on leave today?" -> fromDate = today, toDate = today
    - "Show leaves for yesterday" -> fromDate = yesterday, toDate = yesterday
    - "Who will be on leave tomorrow?" -> fromDate = tomorrow, toDate = tomorrow
    - "Show me 20 leave records" -> pageSize = 20
    
    Returns:
    - Paginated leave history matching the filters
    """
    
    # If prompt is provided, try to extract dates and page size
    if prompt:
        # Extract dates from prompt
        dates = DateParser.extract_dates_from_prompt(prompt)
        if dates["from_date"] and not from_date:
            from_date = dates["from_date"]
        if dates["to_date"] and not to_date:
            to_date = dates["to_date"]
        
        # Extract page size from prompt (e.g., "show me 20 records")
        size_match = re.search(r'(\d+)\s*(record|leave|result)', prompt.lower())
        if size_match:
            page_size = int(size_match.group(1))
    
    # Set default from_date to today if not provided
    if not from_date:
        from_date = datetime.now().strftime("%Y-%m-%d")
    
    # Build payload according to new API structure
    payload = {
        "empId": emp_id,  # null to get all employees
        "reportingManagerId": reporting_manager_id,
        "status": status,
        "pageNumber": page_number,
        "pageSize": page_size,
        "sortOrder": sort_order,
        "sortBy": sort_by,
        "searchTerm": search_term,
        "companyId": company_id,
        "branchId": branch_id,
        "fromDate": from_date,
        "toDate": to_date
    }
    
    logger.info(f"Getting leave history with payload: {payload}")
    return await api_client.get_leave_history(payload)

@mcp.tool()
async def get_leave_statuses() -> Dict[str, Any]:
    """
    Get all possible leave statuses in the system.
    
    Returns:
    - List of leave statuses with their IDs and names
    - Useful for understanding what status values to use in other leave-related tools
    """
    return await api_client.get_leave_statuses()

@mcp.tool()
async def get_leave_types() -> Dict[str, Any]:
    """
    Get all available leave types (e.g., Sick Leave, Casual Leave, Privilege Leave).
    
    Returns:
    - List of leave types with their IDs, names, and properties
    - Use the leave type ID when applying for leave
    
    Note: If this tool fails due to authorization issues, you can use these common leave type IDs:
    - 1: Casual Leave
    - 2: Sick Leave
    - 3: Privilege Leave / Earned Leave
    - 4: Maternity Leave
    - 5: Paternity Leave
    """
    result = await api_client.get_leave_types()
    
    # If the API call fails, return common leave types as fallback
    if result.get("error"):
        logger.warning(f"get_leave_types API failed: {result.get('error')}")
        logger.info("Returning common leave types as fallback")
        return {
            "fallback": True,
            "message": "Could not fetch leave types from API. Using common leave type IDs:",
            "common_leave_types": [
                {"id": 1, "name": "Casual Leave", "description": "For personal or urgent work"},
                {"id": 2, "name": "Sick Leave", "description": "For illness or medical reasons"},
                {"id": 3, "name": "Privilege Leave / Earned Leave", "description": "Annual leave earned by working"},
                {"id": 4, "name": "Maternity Leave", "description": "For maternity purposes"},
                {"id": 5, "name": "Paternity Leave", "description": "For paternity purposes"}
            ],
            "note": "These are common leave type IDs. The actual IDs in your system may differ. You can proceed with leave application using these IDs.",
            "original_error": result.get("error")
        }
    
    return result

@mcp.tool()
async def get_common_leave_type_ids() -> Dict[str, Any]:
    """
    Get a quick reference of common leave type IDs without making an API call.
    Use this when you need to apply leave but get_leave_types fails or is too slow.
    
    Returns:
    - Dictionary of common leave type IDs that work in most HRMS systems
    """
    return {
        "common_leave_types": {
            "casual_leave": {
                "id": 1,
                "name": "Casual Leave",
                "description": "For personal or urgent work, short-term leave",
                "typical_usage": "Personal work, family functions, short errands"
            },
            "sick_leave": {
                "id": 2,
                "name": "Sick Leave",
                "description": "For illness or medical reasons",
                "typical_usage": "When feeling unwell, doctor appointments, medical treatment"
            },
            "privilege_leave": {
                "id": 3,
                "name": "Privilege Leave / Earned Leave",
                "description": "Annual leave earned by working",
                "typical_usage": "Vacations, long trips, extended personal time"
            },
            "maternity_leave": {
                "id": 4,
                "name": "Maternity Leave",
                "description": "For maternity purposes",
                "typical_usage": "Before and after childbirth"
            },
            "paternity_leave": {
                "id": 5,
                "name": "Paternity Leave",
                "description": "For paternity purposes",
                "typical_usage": "After childbirth to support family"
            }
        },
        "usage_instructions": {
            "casual_leave_keywords": ["casual", "personal", "urgent", "family function", "personal work"],
            "sick_leave_keywords": ["sick", "ill", "unwell", "doctor", "medical", "health"],
            "privilege_leave_keywords": ["vacation", "holiday", "annual", "earned", "privilege"]
        },
        "note": "These are typical leave type IDs. Use these when applying leave if get_leave_types fails."
    }

@mcp.tool()
async def get_leave_policies() -> Dict[str, Any]:
    """
    Get all leave policies.
    
    Returns:
    - List of leave policies including accrual rules, carry forward rules, etc.
    """
    return await api_client.get_leave_policies()

@mcp.tool()
async def get_leave_day_parts() -> Dict[str, Any]:
    """
    Get all possible leave day parts.
    
    Returns:
    - List of day parts with their IDs:
      1: First Half
      2: Second Half
      3: Full Day
    """
    return await api_client.get_leave_day_parts()

@mcp.tool()
async def apply_leave(
    leave_type_id: int,
    from_date: str,
    to_date: str,
    leave_reason: str,
    from_leave_day_part: int = 3,
    to_leave_day_part: int = 3,
    number_of_days: Optional[float] = None,
    remarks: Optional[str] = None,
    is_medical_leave: bool = False,
    leave_document: Optional[str] = None,
    emergency_contact: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply for leave. The employee ID is automatically extracted from the authentication token.
    
    IMPORTANT - Common Leave Type IDs (use these if get_leave_types fails):
    - 1: Casual Leave
    - 2: Sick Leave
    - 3: Privilege Leave / Earned Leave
    - 4: Maternity Leave
    - 5: Paternity Leave
    (Use get_leave_types tool to get the complete list and exact IDs for your organization)
    
    Parameters:
    - leave_type_id: ID of the leave type (REQUIRED - use 1 for Casual Leave, 2 for Sick Leave if unsure)
    - from_date: Start date of leave (format: YYYY-MM-DD) (REQUIRED)
    - to_date: End date of leave (format: YYYY-MM-DD) (REQUIRED)
    - leave_reason: Reason for leave (REQUIRED - cannot be empty)
    - from_leave_day_part: Day part for start date (1: First Half, 2: Second Half, 3: Full Day, default: 3)
    - to_leave_day_part: Day part for end date (1: First Half, 2: Second Half, 3: Full Day, default: 3)
    - number_of_days: Number of days (calculated automatically if not provided)
    - remarks: Additional remarks (optional)
    - is_medical_leave: Whether this is medical leave (default: False)
    - leave_document: Document URL/path if any (e.g., medical certificate) (optional)
    - emergency_contact: Emergency contact number during leave (optional)
    
    Note: Employee ID (empId) is automatically extracted from your authentication token.
    You do NOT need to call get_leave_types before applying leave. If you know the leave type:
    - Use 1 for Casual Leave
    - Use 2 for Sick Leave
    - Use 3 for Privilege/Earned Leave
    
    Returns:
    - Success status and leave application details
    """
    
    logger.info("="*60)
    logger.info("APPLY LEAVE TOOL CALLED")
    logger.info(f"Parameters: leave_type_id={leave_type_id}, from_date={from_date}, to_date={to_date}")
    logger.info(f"Leave reason: {leave_reason}")
    logger.info(f"Day parts: from={from_leave_day_part}, to={to_leave_day_part}")
    logger.info("="*60)
    
    # Extract empId from JWT token
    if not AUTH_TOKEN:
        logger.error("No AUTH_TOKEN available in environment")
        return {
            "success": False,
            "error": "No authentication token available. Please ensure the server is properly configured."
        }
    
    logger.info(f"Attempting to extract empId from token (token length: {len(AUTH_TOKEN)})")
    emp_id = JWTHelper.get_emp_id_from_token(AUTH_TOKEN)
    
    if not emp_id:
        logger.error("Failed to extract empId from token")
        return {
            "success": False,
            "error": "Could not extract employee ID from authentication token. Please ensure you are properly authenticated and your token contains an employee ID field."
        }
    
    logger.info(f"✅ Successfully extracted empId={emp_id} from JWT token")
    
    # Validate required fields
    if not leave_reason or leave_reason.strip() == "":
        logger.error("Leave reason is empty or missing")
        return {
            "success": False,
            "error": "Leave reason is required and cannot be empty"
        }
    
    # Calculate number of days if not provided
    if number_of_days is None:
        try:
            start = datetime.strptime(from_date, "%Y-%m-%d")
            end = datetime.strptime(to_date, "%Y-%m-%d")
            days_diff = (end - start).days + 1
            
            # Adjust for half days
            if from_leave_day_part != 3:  # Not full day
                days_diff -= 0.5
            if to_leave_day_part != 3 and from_date != to_date:  # Not full day and different dates
                days_diff -= 0.5
            
            number_of_days = max(0.5, days_diff)  # Minimum 0.5 days
            logger.info(f"Calculated number_of_days: {number_of_days}")
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return {
                "success": False,
                "error": f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
            }
    
    # Validate number_of_days
    if number_of_days < 0.5 or number_of_days > 365:
        logger.error(f"Invalid number_of_days: {number_of_days}")
        return {
            "success": False,
            "error": f"Number of days must be between 0.5 and 365. Calculated: {number_of_days}"
        }
    
    # Validate day parts
    if from_leave_day_part not in [1, 2, 3]:
        logger.error(f"Invalid from_leave_day_part: {from_leave_day_part}")
        return {
            "success": False,
            "error": f"from_leave_day_part must be 1 (First Half), 2 (Second Half), or 3 (Full Day). Got: {from_leave_day_part}"
        }
    
    if to_leave_day_part not in [1, 2, 3]:
        logger.error(f"Invalid to_leave_day_part: {to_leave_day_part}")
        return {
            "success": False,
            "error": f"to_leave_day_part must be 1 (First Half), 2 (Second Half), or 3 (Full Day). Got: {to_leave_day_part}"
        }
    
    # Build leave application payload according to new API structure
    leave_data = {
        "empId": emp_id,
        "leaveTypeId": leave_type_id,
        "fromDate": from_date,
        "fromLeaveDayPart": from_leave_day_part,
        "toDate": to_date,
        "toLeaveDayPart": to_leave_day_part,
        "numberOfDays": number_of_days,
        "leaveReason": leave_reason,
        "remarks": remarks,
        "isMedicalLeave": is_medical_leave,
        "leaveDocument": leave_document,
        "emergencyContact": emergency_contact
    }
    
    logger.info(f"📤 Sending leave application for empId={emp_id}")
    logger.info(f"Payload: {leave_data}")
    
    result = await api_client.apply_leave(leave_data)
    
    if result.get("success"):
        logger.info(f"✅ Leave application successful for empId={emp_id}")
    else:
        logger.error(f"❌ Leave application failed: {result.get('error')}")
    
    return result

# ==== SHIFT TOOLS ====

@mcp.tool()
async def get_assign_shift(
    assign_shift_id: Optional[int] = None,
    search_term: Optional[str] = None,
    employee_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_number: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Get assigned shifts based on filters.
    
    Parameters:
    - assign_shift_id: Specific shift assignment ID
    - search_term: Search term to filter results
    - employee_id: Employee ID to filter shifts
    - from_date: Start date filter (format: YYYY-MM-DD)
    - to_date: End date filter (format: YYYY-MM-DD)
    - page_number: Page number for pagination (default: 1)
    - page_size: Number of records per page (default: 10)
    
    Returns:
    - Collection of assigned shifts matching the filters
    """
    return await api_client.get_assign_shift(
        assign_shift_id=assign_shift_id,
        search_term=search_term,
        employee_id=employee_id,
        from_date=from_date,
        to_date=to_date,
        page_number=page_number,
        page_size=page_size
    )

# ==== ATTENDANCE TOOLS ====

@mcp.tool()
async def get_attendance_policies(
    policy_id: Optional[int] = None,
    is_active: Optional[bool] = None,
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get attendance policies for the company and branch.
    
    Parameters:
    - policy_id: Optional ID of specific policy. If null, lists all policies
    - is_active: Optional filter for active/inactive policies
    - query: Optional query string to filter by policy name
    
    Returns:
    - The requested policy or list of attendance policies
    """
    return await api_client.get_attendance_policies(
        policy_id=policy_id,
        is_active=is_active,
        query=query
    )

# ==== HOLIDAY TOOLS ====

@mcp.tool()
async def get_holidays(
    holiday_type_filter: Optional[str] = None,
    year: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get holidays based on filters.
    
    Parameters:
    - holiday_type_filter: Filter by holiday type
    - year: Filter by year
    - from_date: Start date filter (format: YYYY-MM-DD)
    - to_date: End date filter (format: YYYY-MM-DD)
    
    Returns:
    - Collection of holidays matching the filters
    """
    return await api_client.get_holidays(
        holiday_type_filter=holiday_type_filter,
        year=year,
        from_date=from_date,
        to_date=to_date
    )

# ==== COMPANY TOOLS ====

@mcp.tool()
async def get_company(
    company_id: Optional[str] = None,
    page_no: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Get company information.
    
    Parameters:
    - company_id: Optional company UUID. If provided, gets specific company info
    - page_no: Page number for pagination (default: 1)
    - page_size: Number of records per page (default: 10)
    
    Returns:
    - Company information or list of companies
    """
    return await api_client.get_company(
        company_id=company_id,
        page_no=page_no,
        page_size=page_size
    )

# ==== DEBUG TOOLS ====

@mcp.tool()
async def test_api_connection() -> Dict[str, Any]:
    """
    Test the API connection and authentication.
    Returns the raw response from the leave types endpoint for debugging.
    """
    try:
        headers = api_client._get_headers()
        logger.info(f"Testing API connection with headers: {headers}")
        
        response = await api_client.client.get(
            f"{api_client.base_url}/api/LeaveTypes",
            headers=headers
        )
        
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text,
            "auth_token_present": bool(api_client.auth_token),
            "base_url": api_client.base_url
        }
        
        logger.info(f"API test result: {result}")
        return result
    except Exception as e:
        logger.error(f"API connection test failed: {e}")
        return {
            "error": str(e),
            "auth_token_present": bool(api_client.auth_token),
            "base_url": api_client.base_url
        }

# The transport="stdio" argument tells the server to use standard input/output 
# to receive and respond to tool function calls
if __name__ == "__main__":
    logger.info("Starting HRMS MCP Server...")
    if AUTH_TOKEN:
        logger.info("Auth token received from main.py and loaded successfully")
    else:
        logger.info("MCP server started - auth token will be provided by main.py via environment")
    mcp.run(transport="stdio")