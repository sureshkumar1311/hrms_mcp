import os
import json
import asyncio
import logging
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables 
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tool_calls_made: List[Dict[str, Any]] = []
    follow_up: List[str] = []

class Config:
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    
    # Validate required config
    @classmethod
    def validate(cls):
        if not cls.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
        logger.info(f"Azure OpenAI Endpoint: {cls.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Azure OpenAI Deployment: {cls.AZURE_OPENAI_DEPLOYMENT_NAME}")

config = Config()

class ConversationState:
    """Manages conversation state for dialogue flow"""
    
    def __init__(self):
        self.intent = None  # e.g., "apply_leave", "check_leave_history"
        self.collected_data = {}  # Data collected from user
        self.required_fields = []  # Fields still needed
        self.last_question = None  # Last question asked
        self.stage = None  # Current stage in the flow
    
    def to_dict(self):
        return {
            "intent": self.intent,
            "collected_data": self.collected_data,
            "required_fields": self.required_fields,
            "last_question": self.last_question,
            "stage": self.stage
        }
    
    @classmethod
    def from_dict(cls, data):
        state = cls()
        state.intent = data.get("intent")
        state.collected_data = data.get("collected_data", {})
        state.required_fields = data.get("required_fields", [])
        state.last_question = data.get("last_question")
        state.stage = data.get("stage")
        return state

class MCPServerManager:
    """Manages MCP server for HRMS"""
    
    def __init__(self):
        self.mcp_process = None
        self.is_running = False
        self.current_auth_token = None
    
    def start_mcp_server(self, auth_token: str = ""):
        """Start MCP server as a subprocess with auth token"""
        if self.is_running and self.current_auth_token == auth_token:
            return True
        
        # If server is running with different token, restart it
        if self.is_running and self.current_auth_token != auth_token:
            self.cleanup()
            
        try:
            # Get current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mcp_server_path = os.path.join(current_dir, "hrms_mcp_server.py")
            
            if not os.path.exists(mcp_server_path):
                logger.error(f"MCP server file not found at: {mcp_server_path}")
                return False
            
            # Set up environment with auth token
            env = os.environ.copy()
            env['HRMS_AUTH_TOKEN'] = auth_token
            
            # Start the MCP server
            self.mcp_process = subprocess.Popen(
                [sys.executable, mcp_server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=current_dir,
                env=env
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if process is still running
            if self.mcp_process.poll() is None:
                self.is_running = True
                self.current_auth_token = auth_token
                logger.info(f"MCP server started successfully with PID: {self.mcp_process.pid}")
                return True
            else:
                logger.error("MCP server process terminated immediately")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    def cleanup(self):
        """Clean up MCP server process"""
        if self.mcp_process and self.is_running:
            try:
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=5)
                logger.info("MCP server terminated")
            except:
                try:
                    self.mcp_process.kill()
                except:
                    pass
            finally:
                self.is_running = False
                self.current_auth_token = None

class ChatService:
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.conversation_states: Dict[str, ConversationState] = {}
        self.mcp_client = None
        self.agent = None
        self._initialized = False
        self.server_manager = MCPServerManager()
        self.azure_openai_client = None
        self.available_tools = []
    
    async def initialize_azure_client(self):
        """Initialize Azure OpenAI client if not already done"""
        if self.azure_openai_client is None:
            try:
                # Validate configuration first
                config.validate()
                
                # Initialize Azure OpenAI client
                self.azure_openai_client = AzureChatOpenAI(
                    azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                    api_version=config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                    api_key=config.AZURE_OPENAI_API_KEY,
                    temperature=0.7
                )
                
                # Test Azure OpenAI connection
                test_response = await self.azure_openai_client.ainvoke([{"role": "user", "content": "Hello"}])
                logger.info("Azure OpenAI connection test successful")
                
            except Exception as e:
                logger.error(f"Azure OpenAI connection test failed: {e}")
                raise Exception(f"Azure OpenAI authentication failed: {e}")
    
    async def initialize_with_token(self, auth_token: str):
        """Initialize or reinitialize chat service with auth token"""
        try:
            # Initialize Azure client if needed
            await self.initialize_azure_client()
            
            # Start MCP server with auth token
            if not self.server_manager.start_mcp_server(auth_token):
                raise Exception("Failed to start MCP server")
            
            # Initialize or reinitialize MCP client if token changed
            if (not self._initialized or 
                self.server_manager.current_auth_token != auth_token):
                
                # Close existing client if any
                if self.mcp_client:
                    try:
                        await self.mcp_client.aclose()
                    except:
                        pass
                
                # Initialize MCP client
                current_dir = os.path.dirname(os.path.abspath(__file__))
                mcp_server_path = os.path.join(current_dir, "hrms_mcp_server.py")
                
                self.mcp_client = MultiServerMCPClient({
                    "hrms_management": {
                        "command": "python",
                        "args": [mcp_server_path],
                        "transport": "stdio",
                        "env": {"HRMS_AUTH_TOKEN": auth_token}
                    }
                })
                
                # Get tools from MCP server
                tools = await self.mcp_client.get_tools()
                self.available_tools = tools
                logger.info(f"Loaded {len(tools)} tools from MCP server: {[tool.name for tool in tools]}")
                
                self.agent = create_react_agent(
                    self.azure_openai_client, 
                    tools,
                )
                self._initialized = True
                logger.info("Chat service initialized successfully with new auth token")
            
        except Exception as e:
            logger.error(f"Error initializing chat service: {e}")
            # Cleanup on failure
            self.server_manager.cleanup()
            raise
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self.conversations.get(conversation_id, [])
    
    def get_conversation_state(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = ConversationState()
        return self.conversation_states[conversation_id]
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)
    
    def detect_intent(self, message: str) -> Optional[str]:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        # Leave application intents
        if any(word in message_lower for word in ["apply leave", "apply for leave", "request leave", "take leave", "need leave", "want leave"]):
            return "apply_leave"
        
        # Leave history/check intents
        if any(word in message_lower for word in ["leave history", "my leaves", "check leave", "leave status", "who is on leave", "who's on leave"]):
            return "check_leave_history"
        
        # Get leave types
        if any(word in message_lower for word in ["leave types", "types of leave", "what leaves", "available leaves"]):
            return "get_leave_types"
        
        return None
    
    async def get_follow_up_suggestions(self, intent: str, state: ConversationState, last_response: str) -> List[str]:
        """Generate contextual follow-up suggestions"""
        follow_ups = []
        
        if intent == "apply_leave":
            if state.stage == "ask_leave_type":
                # Get leave types from tools
                try:
                    leave_types_tool = next((tool for tool in self.available_tools if tool.name == "get_common_leave_type_ids"), None)
                    if leave_types_tool:
                        result = await self.mcp_client.call_tool("get_common_leave_type_ids", {})
                        if result and "common_leave_types" in result:
                            follow_ups = [lt["name"] for lt in result["common_leave_types"].values()][:5]
                    else:
                        follow_ups = ["Casual Leave", "Sick Leave", "Privilege Leave"]
                except Exception as e:
                    logger.error(f"Error getting leave types for follow-up: {e}")
                    follow_ups = ["Casual Leave", "Sick Leave", "Privilege Leave"]
            
            elif state.stage == "ask_dates":
                follow_ups = ["Today", "Tomorrow", "Next Monday", "Specify custom dates"]
            
            elif state.stage == "ask_reason":
                if "sick" in state.collected_data.get("leave_type", "").lower():
                    follow_ups = ["Fever", "Doctor appointment", "Medical checkup", "Not feeling well"]
                elif "casual" in state.collected_data.get("leave_type", "").lower():
                    follow_ups = ["Personal work", "Family function", "Emergency", "Other personal reason"]
            
            elif state.stage == "ask_day_part":
                follow_ups = ["Full Day", "First Half", "Second Half"]
            
            elif state.stage == "confirm":
                follow_ups = ["Yes, apply", "No, cancel", "Modify details"]
        
        elif intent == "check_leave_history":
            if state.stage == "ask_time_period":
                follow_ups = ["Today", "This week", "This month", "Last 30 days", "Custom date range"]
            
            elif state.stage == "ask_employee":
                follow_ups = ["My leaves", "All employees", "My team"]
        
        return follow_ups
    
    async def handle_dialogue_flow(self, message: str, conversation_id: str, state: ConversationState) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """Handle multi-turn dialogue flow for collecting information"""
        
        tool_calls_made = []
        follow_ups = []
        
        # Detect intent if not already set
        if not state.intent:
            state.intent = self.detect_intent(message)
        
        # Handle apply_leave flow
        if state.intent == "apply_leave":
            # Stage 1: Ask for leave type
            if not state.collected_data.get("leave_type"):
                if not state.stage:
                    state.stage = "ask_leave_type"
                    # Call get_common_leave_type_ids tool
                    try:
                        result = await self.mcp_client.call_tool("get_common_leave_type_ids", {})
                        tool_calls_made.append({
                            "tool_name": "get_common_leave_type_ids",
                            "parameters": {},
                            "result": "executed"
                        })
                        
                        if result and "common_leave_types" in result:
                            leave_types_list = []
                            follow_ups = []
                            for lt_key, lt_data in result["common_leave_types"].items():
                                leave_types_list.append(f"**{lt_data['name']}** (ID: {lt_data['id']}): {lt_data['description']}")
                                follow_ups.append(lt_data['name'])
                            
                            response = f"I'll help you apply for leave. First, please choose the type of leave:\n\n" + "\n".join(leave_types_list)
                            return response, follow_ups[:5], tool_calls_made
                    except Exception as e:
                        logger.error(f"Error calling get_common_leave_type_ids: {e}")
                    
                    response = "I'll help you apply for leave. What type of leave would you like to apply for? (e.g., Casual Leave, Sick Leave, Privilege Leave)"
                    follow_ups = ["Casual Leave", "Sick Leave", "Privilege Leave"]
                    return response, follow_ups, tool_calls_made
                else:
                    # Extract leave type from response
                    message_lower = message.lower()
                    leave_type_map = {
                        "casual": {"id": 1, "name": "Casual Leave"},
                        "sick": {"id": 2, "name": "Sick Leave"},
                        "privilege": {"id": 3, "name": "Privilege Leave"},
                        "earned": {"id": 3, "name": "Privilege Leave"},
                        "maternity": {"id": 4, "name": "Maternity Leave"},
                        "paternity": {"id": 5, "name": "Paternity Leave"}
                    }
                    
                    for key, value in leave_type_map.items():
                        if key in message_lower:
                            state.collected_data["leave_type"] = value["name"]
                            state.collected_data["leave_type_id"] = value["id"]
                            state.stage = "ask_dates"
                            break
                    
                    if not state.collected_data.get("leave_type"):
                        response = "I couldn't understand the leave type. Please choose from: Casual Leave, Sick Leave, or Privilege Leave."
                        follow_ups = ["Casual Leave", "Sick Leave", "Privilege Leave"]
                        return response, follow_ups, tool_calls_made
            
            # Stage 2: Ask for dates
            if not state.collected_data.get("from_date"):
                if state.stage == "ask_dates":
                    response = f"Great! You've selected **{state.collected_data['leave_type']}**. When would you like to take this leave? Please provide the start date (or say 'today', 'tomorrow', etc.)"
                    follow_ups = ["Today", "Tomorrow", "Day after tomorrow"]
                    return response, follow_ups, tool_calls_made
                else:
                    # Parse date from message
                    from datetime import datetime, timedelta
                    message_lower = message.lower()
                    
                    if "today" in message_lower:
                        state.collected_data["from_date"] = datetime.now().strftime("%Y-%m-%d")
                        state.collected_data["to_date"] = datetime.now().strftime("%Y-%m-%d")
                    elif "tomorrow" in message_lower:
                        tomorrow = datetime.now() + timedelta(days=1)
                        state.collected_data["from_date"] = tomorrow.strftime("%Y-%m-%d")
                        state.collected_data["to_date"] = tomorrow.strftime("%Y-%m-%d")
                    else:
                        # Try to extract date in YYYY-MM-DD format
                        import re
                        date_match = re.search(r'\d{4}-\d{2}-\d{2}', message)
                        if date_match:
                            state.collected_data["from_date"] = date_match.group(0)
                            state.collected_data["to_date"] = date_match.group(0)
                    
                    if not state.collected_data.get("from_date"):
                        response = "I couldn't understand the date. Please provide a date in YYYY-MM-DD format, or say 'today', 'tomorrow', etc."
                        follow_ups = ["Today", "Tomorrow"]
                        return response, follow_ups, tool_calls_made
                    
                    state.stage = "ask_end_date"
            
            # Stage 3: Ask for end date if different
            if state.collected_data.get("from_date") and not state.collected_data.get("end_date_confirmed"):
                if state.stage == "ask_end_date":
                    response = f"Your leave will start on **{state.collected_data['from_date']}**. Is this a single day leave, or do you need multiple days? If multiple days, please provide the end date."
                    follow_ups = ["Single day", "Multiple days"]
                    return response, follow_ups, tool_calls_made
                else:
                    message_lower = message.lower()
                    if "single" in message_lower or "one day" in message_lower or "same" in message_lower:
                        state.collected_data["end_date_confirmed"] = True
                        state.stage = "ask_day_part"
                    else:
                        # Try to extract end date
                        import re
                        from datetime import datetime, timedelta
                        
                        date_match = re.search(r'\d{4}-\d{2}-\d{2}', message)
                        if date_match:
                            state.collected_data["to_date"] = date_match.group(0)
                            state.collected_data["end_date_confirmed"] = True
                            state.stage = "ask_day_part"
                        elif "days" in message_lower:
                            # Extract number of days
                            days_match = re.search(r'(\d+)\s*days?', message_lower)
                            if days_match:
                                num_days = int(days_match.group(1))
                                from_date_obj = datetime.strptime(state.collected_data["from_date"], "%Y-%m-%d")
                                to_date_obj = from_date_obj + timedelta(days=num_days - 1)
                                state.collected_data["to_date"] = to_date_obj.strftime("%Y-%m-%d")
                                state.collected_data["end_date_confirmed"] = True
                                state.stage = "ask_day_part"
                    
                    if not state.collected_data.get("end_date_confirmed"):
                        response = "Please specify if this is a single day leave or provide the end date."
                        follow_ups = ["Single day", "2 days", "3 days"]
                        return response, follow_ups, tool_calls_made
            
            # Stage 4: Ask for day part
            if not state.collected_data.get("day_part"):
                if state.stage == "ask_day_part":
                    response = "Do you need a full day leave or half day?"
                    follow_ups = ["Full Day", "First Half", "Second Half"]
                    return response, follow_ups, tool_calls_made
                else:
                    message_lower = message.lower()
                    if "full" in message_lower:
                        state.collected_data["day_part"] = 3
                        state.collected_data["day_part_name"] = "Full Day"
                    elif "first" in message_lower or "morning" in message_lower:
                        state.collected_data["day_part"] = 1
                        state.collected_data["day_part_name"] = "First Half"
                    elif "second" in message_lower or "afternoon" in message_lower:
                        state.collected_data["day_part"] = 2
                        state.collected_data["day_part_name"] = "Second Half"
                    
                    if not state.collected_data.get("day_part"):
                        response = "Please choose: Full Day, First Half, or Second Half"
                        follow_ups = ["Full Day", "First Half", "Second Half"]
                        return response, follow_ups, tool_calls_made
                    
                    state.stage = "ask_reason"
            
            # Stage 5: Ask for reason
            if not state.collected_data.get("reason"):
                if state.stage == "ask_reason":
                    response = "Please provide a reason for your leave:"
                    
                    if "sick" in state.collected_data.get("leave_type", "").lower():
                        follow_ups = ["Fever", "Doctor appointment", "Medical checkup", "Not feeling well"]
                    elif "casual" in state.collected_data.get("leave_type", "").lower():
                        follow_ups = ["Personal work", "Family function", "Emergency"]
                    else:
                        follow_ups = []
                    
                    return response, follow_ups, tool_calls_made
                else:
                    state.collected_data["reason"] = message
                    state.stage = "confirm"
            
            # Stage 6: Confirm and apply
            if state.stage == "confirm":
                summary = f"""
Please confirm your leave application:

**Leave Type:** {state.collected_data['leave_type']}
**From Date:** {state.collected_data['from_date']} ({state.collected_data.get('day_part_name', 'Full Day')})
**To Date:** {state.collected_data['to_date']}
**Reason:** {state.collected_data['reason']}

Should I proceed with this application?
"""
                follow_ups = ["Yes, apply", "No, cancel"]
                state.stage = "awaiting_confirmation"
                return summary, follow_ups, tool_calls_made
            
            # Stage 7: Execute application
            if state.stage == "awaiting_confirmation":
                message_lower = message.lower()
                if "yes" in message_lower or "apply" in message_lower or "proceed" in message_lower or "confirm" in message_lower:
                    # Apply leave using the tool
                    try:
                        result = await self.mcp_client.call_tool("apply_leave", {
                            "leave_type_id": state.collected_data["leave_type_id"],
                            "from_date": state.collected_data["from_date"],
                            "to_date": state.collected_data["to_date"],
                            "leave_reason": state.collected_data["reason"],
                            "from_leave_day_part": state.collected_data.get("day_part", 3),
                            "to_leave_day_part": state.collected_data.get("day_part", 3)
                        })
                        
                        tool_calls_made.append({
                            "tool_name": "apply_leave",
                            "parameters": state.collected_data,
                            "result": "executed"
                        })
                        
                        # Reset state
                        state.intent = None
                        state.collected_data = {}
                        state.stage = None
                        
                        if result.get("success"):
                            return "✅ Your leave application has been submitted successfully!", [], tool_calls_made
                        else:
                            return f"❌ Failed to apply leave: {result.get('message', 'Unknown error')}", [], tool_calls_made
                    except Exception as e:
                        logger.error(f"Error applying leave: {e}")
                        return f"❌ Error applying leave: {str(e)}", [], tool_calls_made
                else:
                    # Reset state
                    state.intent = None
                    state.collected_data = {}
                    state.stage = None
                    return "Leave application cancelled. How else can I help you?", [], tool_calls_made
        
        # Handle check_leave_history flow
        elif state.intent == "check_leave_history":
            if not state.collected_data.get("time_period"):
                if not state.stage:
                    state.stage = "ask_time_period"
                    response = "What time period would you like to check?"
                    follow_ups = ["Today", "This week", "This month", "Last 30 days"]
                    return response, follow_ups, tool_calls_made
                else:
                    # Parse time period
                    from datetime import datetime, timedelta
                    message_lower = message.lower()
                    
                    if "today" in message_lower:
                        state.collected_data["from_date"] = datetime.now().strftime("%Y-%m-%d")
                        state.collected_data["to_date"] = datetime.now().strftime("%Y-%m-%d")
                    elif "this week" in message_lower:
                        today = datetime.now()
                        start_of_week = today - timedelta(days=today.weekday())
                        end_of_week = start_of_week + timedelta(days=6)
                        state.collected_data["from_date"] = start_of_week.strftime("%Y-%m-%d")
                        state.collected_data["to_date"] = end_of_week.strftime("%Y-%m-%d")
                    elif "this month" in message_lower:
                        today = datetime.now()
                        start_of_month = today.replace(day=1)
                        state.collected_data["from_date"] = start_of_month.strftime("%Y-%m-%d")
                        state.collected_data["to_date"] = today.strftime("%Y-%m-%d")
                    
                    if state.collected_data.get("from_date"):
                        state.collected_data["time_period"] = message_lower
                        state.stage = "execute"
            
            # Execute the query
            if state.stage == "execute":
                try:
                    result = await self.mcp_client.call_tool("get_leave_history", {
                        "from_date": state.collected_data.get("from_date"),
                        "to_date": state.collected_data.get("to_date"),
                        "page_size": 20
                    })
                    
                    tool_calls_made.append({
                        "tool_name": "get_leave_history",
                        "parameters": state.collected_data,
                        "result": "executed"
                    })
                    
                    # Reset state
                    state.intent = None
                    state.collected_data = {}
                    state.stage = None
                    
                    # Format the response
                    if result and isinstance(result, dict) and "data" in result:
                        leaves = result["data"]
                        if leaves:
                            response = f"Here are the leaves for {state.collected_data.get('time_period', 'the selected period')}:\n\n"
                            for leave in leaves[:10]:  # Show first 10
                                response += f"• {leave.get('employeeName', 'Unknown')} - {leave.get('leaveType', 'Unknown')} ({leave.get('fromDate')} to {leave.get('toDate')})\n"
                            return response, [], tool_calls_made
                        else:
                            return "No leaves found for the selected period.", [], tool_calls_made
                    else:
                        return "Unable to fetch leave history at the moment.", [], tool_calls_made
                except Exception as e:
                    logger.error(f"Error fetching leave history: {e}")
                    return f"Error fetching leave history: {str(e)}", [], tool_calls_made
        
        # Default: no dialogue flow active
        return None, [], tool_calls_made
    
    async def process_chat(self, message: str, conversation_id: str, auth_token: str) -> ChatResponse:
        """Process chat message with auth token and dialogue flow"""
        
        # Ensure initialization with current auth token
        if not auth_token:
            raise HTTPException(status_code=401, detail="Authorization token is required")
        
        await self.initialize_with_token(auth_token)
        
        try:
            # Get conversation history and state
            conversation_history = self.get_conversation(conversation_id)
            state = self.get_conversation_state(conversation_id)
            
            # Try dialogue flow first
            dialogue_response, follow_ups, tool_calls_made = await self.handle_dialogue_flow(message, conversation_id, state)
            
            if dialogue_response:
                # Update conversation history
                self.add_message(conversation_id, {"role": "user", "content": message})
                self.add_message(conversation_id, {"role": "assistant", "content": dialogue_response})
                
                return ChatResponse(
                    response=dialogue_response,
                    conversation_id=conversation_id,
                    tool_calls_made=tool_calls_made,
                    follow_up=follow_ups
                )
            
            # Fall back to agent-based processing
            messages = []
            for msg in conversation_history:
                messages.append(msg)
            messages.append({"role": "user", "content": message})
            
            # Add system message for context
            system_msg = """You are an HRMS assistant. Help users with leave management, attendance, and other HR-related queries. Be concise and helpful."""
            
            # Process with LangGraph agent
            response = await self.agent.ainvoke({"messages": messages})
            
            # Extract the assistant's response
            assistant_message = response['messages'][-1].content
            
            # Update conversation history
            self.add_message(conversation_id, {"role": "user", "content": message})
            self.add_message(conversation_id, {"role": "assistant", "content": assistant_message})
            
            # Extract tool calls information
            agent_tool_calls = []
            for msg in response['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        agent_tool_calls.append({
                            "tool_name": tool_call.get('name', 'unknown'),
                            "parameters": tool_call.get('args', {}),
                            "result": "executed"
                        })
            
            return ChatResponse(
                response=assistant_message,
                conversation_id=conversation_id,
                tool_calls_made=agent_tool_calls,
                follow_up=[]
            )
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            return ChatResponse(
                response=error_response,
                conversation_id=conversation_id,
                tool_calls_made=[],
                follow_up=[]
            )
    
    async def cleanup(self):
        """Clean up resources"""
        if self.mcp_client:
            try:
                await self.mcp_client.aclose()
            except:
                pass
        self.server_manager.cleanup()

# Initialize chat service
chat_service = ChatService()

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting HRMS Chat API...")
    try:
        await chat_service.initialize_azure_client()
        logger.info("Azure OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure client: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HRMS Chat API...")
    await chat_service.cleanup()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="HRMS Chat API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme for FastAPI docs
security = HTTPBearer(
    scheme_name="Bearer Token",
    description="Enter your HRMS auth token"
)

# Dependency to extract auth token from header
async def get_auth_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract auth token from Authorization header"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    
    return credentials.credentials

# ------------------ API ENDPOINTS ------------------ #

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, auth_token: str = Depends(get_auth_token)):
    """Main chat endpoint with dynamic auth token and dialogue flow"""
    try:
        conversation_id = request.conversation_id or f"conv_{len(chat_service.conversations) + 1}"
        response = await chat_service.process_chat(request.message, conversation_id, auth_token)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/tools")
async def get_available_tools(auth_token: str = Depends(get_auth_token)):
    """Get available tools with auth token"""
    try:
        await chat_service.initialize_with_token(auth_token)
        tools = await chat_service.mcp_client.get_tools()
        return {"tools": [{"name": tool.name, "description": tool.description} for tool in tools]}
    except Exception as e:
        return {"error": f"Could not fetch tools: {str(e)}", "tools": []}

@app.post("/reset-conversation")
async def reset_conversation(conversation_id: str, auth_token: str = Depends(get_auth_token)):
    """Reset a conversation and its state"""
    try:
        if conversation_id in chat_service.conversations:
            del chat_service.conversations[conversation_id]
        if conversation_id in chat_service.conversation_states:
            del chat_service.conversation_states[conversation_id]
        return {"message": "Conversation reset successfully", "conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting conversation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)), log_level="info")