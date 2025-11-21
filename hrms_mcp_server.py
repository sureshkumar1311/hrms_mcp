import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
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

def get_current_auth_token() -> str:
    """Get the current auth token from environment - read fresh each time"""
    token = os.getenv("HRMS_AUTH_TOKEN", "")
    if not token:
        logger.warning("No auth token found in environment variable HRMS_AUTH_TOKEN")
    return token

class JWTHelper:
    """Helper class to decode JWT tokens"""
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode JWT token without verification"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            logger.info(f"Decoded JWT payload keys: {list(payload.keys())}")
            if 'uid' in payload:
                logger.info(f"Found uid in token: {payload['uid']}")
            return payload
        except Exception as e:
            logger.error(f"Failed to decode JWT: {e}")
            return {}
    
    @staticmethod
    def get_emp_id_from_token(token: str) -> Optional[int]:
        """Extract empId from JWT token - looks for 'uid' field"""
        if not token:
            logger.error("No token provided to get_emp_id_from_token")
            return None
            
        logger.info(f"Extracting empId from token (length: {len(token)})")
        
        decoded = JWTHelper.decode_token(token)
        if decoded:
            # Check 'uid' field first as per your HRMS system
            for key in ['uid', 'empId', 'emp_id', 'employeeId', 'employee_id', 'sub', 'userId', 'user_id', 'id']:
                if key in decoded:
                    try:
                        emp_id = int(decoded[key])
                        logger.info(f" Successfully extracted empId={emp_id} from JWT field '{key}'")
                        return emp_id
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Found key '{key}' but couldn't convert to int: {decoded[key]}")
                        continue
            logger.error(f" Could not find empId in token. Available keys: {list(decoded.keys())}")
        else:
            logger.error(" Token decoding returned empty dict")
        return None

class HRMSAPIClient:
    """Client to interact with the HRMS APIs"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"HRMSAPIClient initialized with base URL: {base_url}")
    
    def _get_headers(self, auth_token: str = None) -> Dict[str, str]:
        """Get headers with fresh auth token"""
        if not auth_token:
            auth_token = get_current_auth_token()
        
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            logger.info(f"Using auth token (first 20 chars): {auth_token[:20]}...")
        else:
            logger.warning(" No auth token available for headers")
        return headers
    
    async def get_leave_history(self, payload: Dict[str, Any]):
        """Get leave history using POST request with payload"""
        try:
            logger.info(f" Fetching leave history with payload: {json.dumps(payload, indent=2)}")
            response = await self.client.post(
                f"{self.base_url}/api/Leave/history",
                json=payload,
                headers=self._get_headers()
            )
            
            logger.info(f"Leave history API response status: {response.status_code}")
            
            if response.status_code == 401:
                logger.error(" 401 Unauthorized - Auth token invalid")
                return {"error": "Unauthorized", "status_code": 401}
            
            response.raise_for_status()
            result = response.json()
            logger.info(f" Leave history fetched successfully. Records: {len(result.get('data', []))}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f" HTTP error fetching leave history: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            logger.error(f" Error fetching leave history: {e}")
            return {"error": str(e)}
    
    async def get_leave_statuses(self):
        """Get all possible leave statuses"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/Leave/statuses",
                headers=self._get_headers()
            )
            logger.info(f"Leave statuses response status: {response.status_code}")
            
            if response.status_code == 401:
                logger.error("401 Unauthorized")
                return {"error": "Unauthorized", "status_code": 401}
            
            response.raise_for_status()
            result = response.json()
            logger.info(f" Leave statuses fetched successfully")
            return result
        except Exception as e:
            logger.error(f"Error fetching leave statuses: {e}")
            return {"error": str(e)}
    
    async def get_leave_types(self):
        """Get all leave types from API"""
        try:
            logger.info(" Fetching leave types from API...")
            response = await self.client.get(
                f"{self.base_url}/api/LeaveTypes",
                headers=self._get_headers()
            )
            logger.info(f"Leave types response status: {response.status_code}")
            
            if response.status_code == 401:
                logger.error(" 401 Unauthorized")
                return {"error": "Unauthorized", "status_code": 401}
            
            response.raise_for_status()
            result = response.json()
            logger.info(f" Leave types fetched successfully: {len(result) if isinstance(result, list) else 'N/A'} types")
            return {"leave_types": result, "success": True}
        except Exception as e:
            logger.error(f" Error fetching leave types: {e}")
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
    
    async def apply_leave(self, leave_data: Dict[str, Any]):
        """Apply for leave with proper payload structure"""
        try:
            current_token = get_current_auth_token()
            
            logger.info("="*80)
            logger.info(" API CLIENT: Applying leave")
            logger.info(f"URL: {self.base_url}/api/Leave/apply")
            logger.info(f"Payload: {json.dumps(leave_data, indent=2)}")
            logger.info(f"Auth token present: {bool(current_token)}")
            logger.info("="*80)
            
            response = await self.client.post(
                f"{self.base_url}/api/Leave/apply",
                json=leave_data,
                headers=self._get_headers(current_token)
            )
            
            logger.info(f"API Response Status: {response.status_code}")
            logger.info(f"API Response Body: {response.text}")
            
            if response.status_code == 401:
                logger.error(" 401 Unauthorized - Token invalid")
                return {
                    "success": False,
                    "error": "Unauthorized",
                    "message": "Authentication token is invalid or expired"
                }
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f" Leave application successful for empId={leave_data['empId']}!")
            
            return {"success": True, "data": result, "message": "Leave application submitted successfully"}
            
        except httpx.HTTPStatusError as e:
            logger.error("="*80)
            logger.error(f" HTTP error applying leave")
            logger.error(f"Status Code: {e.response.status_code}")
            logger.error(f"Response Body: {e.response.text}")
            logger.error("="*80)
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "details": e.response.text,
                "message": f"Failed to apply leave. Status: {e.response.status_code}"
            }
        except Exception as e:
            logger.error("="*80)
            logger.error(f" Exception applying leave: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error("="*80)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to apply leave: {str(e)}"
            }

# Initialize FastMCP server and API client
mcp = FastMCP("hrms-management")
api_client = HRMSAPIClient(HRMS_API_BASE_URL)

# Import json for logging
import json

# ==== LEAVE TOOLS ====

@mcp.tool()
async def get_leave_history(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    emp_id: Optional[int] = None,
    status: Optional[int] = None,
    page_number: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Get leave history based on filters.
    
    Parameters:
    - from_date: Start date (YYYY-MM-DD format). Defaults to today if not provided.
    - to_date: End date (YYYY-MM-DD format). Optional.
    - emp_id: Employee ID filter. Null to get all employees.
    - status: Leave status filter. Null for all statuses.
    - page_number: Page number (default: 1)
    - page_size: Records per page (default: 10)
    
    Returns:
    - Paginated leave history
    """
    
    # Set default from_date to today if not provided
    if not from_date:
        from_date = datetime.now().strftime("%Y-%m-%d")
    
    # Build payload
    payload = {
        "empId": emp_id,
        "reportingManagerId": None,
        "status": status,
        "pageNumber": page_number,
        "pageSize": page_size,
        "sortOrder": None,
        "sortBy": None,
        "searchTerm": None,
        "companyId": None,
        "branchId": None,
        "fromDate": from_date,
        "toDate": to_date
    }
    
    logger.info(f" Getting leave history with payload: {json.dumps(payload, indent=2)}")
    return await api_client.get_leave_history(payload)

@mcp.tool()
async def get_leave_statuses() -> Dict[str, Any]:
    """
    Get all possible leave statuses in the system.
    
    Returns:
    - List of leave statuses with their IDs and names
    """
    return await api_client.get_leave_statuses()

@mcp.tool()
async def get_leave_types() -> Dict[str, Any]:
    """
    Get all available leave types from the HRMS API.
    
    Returns:
    - List of leave types with their IDs, names, and properties
    - Use the leave type ID when applying for leave
    
    Note: This calls the actual API to get real leave types configured in your system.
    """
    result = await api_client.get_leave_types()
    
    # If API call fails, return common fallback types
    if isinstance(result, dict) and result.get("error"):
        logger.warning(f"API failed, returning fallback leave types")
        return {
            "fallback": True,
            "message": "Using common leave types as API call failed",
            "common_leave_types": [
                {"id": 1, "name": "Casual Leave", "description": "For personal or urgent work"},
                {"id": 2, "name": "Sick Leave", "description": "For illness or medical reasons"}
            ],
            "original_error": result.get("error")
        }
    
    return result

@mcp.tool()
async def get_leave_day_parts() -> Dict[str, Any]:
    """
    Get all possible leave day parts.
    
    Returns:
    - List of day parts:
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
    Apply for leave. Employee ID is automatically extracted from the JWT token.
    
    IMPORTANT: empId is extracted from the 'uid' field in your JWT token.
    
    Parameters:
    - leave_type_id: ID of the leave type (REQUIRED)
    - from_date: Start date (YYYY-MM-DD) (REQUIRED)
    - to_date: End date (YYYY-MM-DD) (REQUIRED)
    - leave_reason: Reason for leave (REQUIRED - cannot be empty)
    - from_leave_day_part: Day part for start (1=First Half, 2=Second Half, 3=Full Day, default: 3)
    - to_leave_day_part: Day part for end (1=First Half, 2=Second Half, 3=Full Day, default: 3)
    - number_of_days: Number of days (auto-calculated if not provided)
    - remarks: Additional remarks (optional)
    - is_medical_leave: Whether medical leave (default: False)
    - leave_document: Document URL if any (optional)
    - emergency_contact: Emergency contact number (optional)
    
    Returns:
    - Success status and leave application details
    """
    
    logger.info("="*80)
    logger.info(" APPLY LEAVE TOOL CALLED")
    logger.info(f"Parameters received:")
    logger.info(f"  - leave_type_id: {leave_type_id}")
    logger.info(f"  - from_date: {from_date}")
    logger.info(f"  - to_date: {to_date}")
    logger.info(f"  - leave_reason: {leave_reason}")
    logger.info(f"  - from_leave_day_part: {from_leave_day_part}")
    logger.info(f"  - to_leave_day_part: {to_leave_day_part}")
    logger.info("="*80)
    
    # Get FRESH auth token
    current_token = get_current_auth_token()
    
    if not current_token:
        logger.error(" No AUTH_TOKEN available in environment")
        return {
            "success": False,
            "error": "No authentication token available",
            "message": "Server configuration error: No authentication token"
        }
    
    logger.info(f" Current token length: {len(current_token)}")
    
    # Extract empId from JWT token
    emp_id = JWTHelper.get_emp_id_from_token(current_token)
    
    if not emp_id:
        logger.error(" Failed to extract empId from token")
        return {
            "success": False,
            "error": "Could not extract employee ID from authentication token",
            "message": "Authentication error: Could not extract employee ID from token"
        }
    
    logger.info(f" Successfully extracted empId={emp_id} from JWT token")
    
    # Validate required fields
    if not leave_reason or leave_reason.strip() == "":
        logger.error(" Leave reason is empty")
        return {
            "success": False,
            "error": "Leave reason is required and cannot be empty",
            "message": "Validation error: Leave reason is required"
        }
    
    # Calculate number of days if not provided
    if number_of_days is None:
        try:
            start = datetime.strptime(from_date, "%Y-%m-%d")
            end = datetime.strptime(to_date, "%Y-%m-%d")
            days_diff = (end - start).days + 1
            
            # Adjust for half days
            if from_leave_day_part != 3:
                days_diff -= 0.5
            if to_leave_day_part != 3 and from_date != to_date:
                days_diff -= 0.5
            
            number_of_days = max(0.5, days_diff)
            logger.info(f" Calculated number_of_days: {number_of_days}")
        except ValueError as e:
            logger.error(f" Date parsing error: {e}")
            return {
                "success": False,
                "error": f"Invalid date format: {str(e)}",
                "message": "Validation error: Invalid date format. Use YYYY-MM-DD"
            }
    
    # Validate number_of_days
    if number_of_days < 0.5 or number_of_days > 365:
        logger.error(f" Invalid number_of_days: {number_of_days}")
        return {
            "success": False,
            "error": f"Number of days must be between 0.5 and 365. Got: {number_of_days}",
            "message": f"Validation error: Invalid number of days ({number_of_days})"
        }
    
    # Build leave application payload
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
    logger.info(f"Full Payload: {json.dumps(leave_data, indent=2)}")
    
    result = await api_client.apply_leave(leave_data)
    
    if result.get("success"):
        logger.info(f" Leave application successful for empId={emp_id}")
    else:
        logger.error(f" Leave application failed for empId={emp_id}: {result.get('error')}")
    
    return result

if __name__ == "__main__":
    logger.info(" Starting HRMS MCP Server...")
    current_token = get_current_auth_token()
    if current_token:
        logger.info(f" Auth token present at startup (length: {len(current_token)})")
        # Try to decode to verify
        emp_id = JWTHelper.get_emp_id_from_token(current_token)
        if emp_id:
            logger.info(f" Token is valid. Employee ID: {emp_id}")
        else:
            logger.warning(" Could not extract employee ID from token")
    else:
        logger.warning(" No auth token at startup - will be read fresh on each request")
    mcp.run(transport="stdio")