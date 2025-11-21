import os
import json
import asyncio
import logging
import subprocess
import sys
import time
import re
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
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
    follow_up: Optional[List[str]] = None

class Config:
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    @classmethod
    def validate(cls):
        if not cls.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
        logger.info(f"Azure OpenAI Endpoint: {cls.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Azure OpenAI Deployment: {cls.AZURE_OPENAI_DEPLOYMENT_NAME}")

config = Config()

class MCPServerManager:
    """Manages MCP server for HRMS"""
    
    def __init__(self):
        self.mcp_process = None
        self.is_running = False
        self.current_auth_token = None
    
    def start_mcp_server(self, auth_token: str = ""):
        """Start MCP server as a subprocess with auth token"""
        if self.is_running and self.current_auth_token == auth_token:
            logger.info("MCP server already running with same token")
            return True
        
        # If server is running with different token, restart it
        if self.is_running and self.current_auth_token != auth_token:
            logger.info("Restarting MCP server with new token")
            self.cleanup()
            
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mcp_server_path = os.path.join(current_dir, "hrms_mcp_server.py")
            
            if not os.path.exists(mcp_server_path):
                logger.error(f"MCP server file not found at: {mcp_server_path}")
                return False
            
            # Set up environment with auth token
            env = os.environ.copy()
            env['HRMS_AUTH_TOKEN'] = auth_token
            
            logger.info(f"Starting MCP server subprocess...")
            
            # Start the MCP server
            self.mcp_process = subprocess.Popen(
                [sys.executable, mcp_server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=current_dir,
                env=env
            )
            
            # Give it time to start
            time.sleep(2)
            
            # Check if process is still running
            if self.mcp_process.poll() is None:
                self.is_running = True
                self.current_auth_token = auth_token
                logger.info(f"✅ MCP server started successfully with PID: {self.mcp_process.pid}")
                return True
            else:
                stderr_output = self.mcp_process.stderr.read() if self.mcp_process.stderr else b""
                logger.error(f"❌ MCP server process terminated immediately")
                logger.error(f"stderr: {stderr_output.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
                    logger.info("MCP server killed")
                except:
                    pass
            finally:
                self.is_running = False
                self.current_auth_token = None

class ChatService:
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.mcp_client = None
        self.agent = None
        self._initialized = False
        self.server_manager = MCPServerManager()
        self.azure_openai_client = None
        self.available_tools = []
        
        # System prompt that guides the LLM with follow-up suggestions
        self.system_prompt = """You are an intelligent HRMS (Human Resource Management System) assistant. You help employees with leave management and HR queries.

**Your Capabilities:**
You have access to several tools to help users:
- apply_leave: Apply for leave (requires leave_type_id, from_date, to_date, leave_reason)
- get_leave_history: Check leave history and who is on leave
- get_leave_types: Get available leave types
- get_leave_statuses: Get leave status options
- get_holidays: Check company holidays

**Guidelines:**
1. **Be conversational and natural** - Have a friendly, helpful conversation with users
2. **Ask clarifying questions** - If information is missing, ask the user naturally
3. **Use tools intelligently** - Call the appropriate tool when you have enough information
4. **Handle context switches** - If user changes topic mid-conversation (e.g., from applying leave to checking history), switch context immediately
5. **Parse dates flexibly** - Understand "today", "tomorrow", "December 29", "29th Dec", etc.
6. **Common leave types**:
   - Leave Type ID 1 = Casual Leave (for personal work)
   - Leave Type ID 2 = Sick Leave (for medical reasons)
   - Leave Type ID 3 = Privilege Leave (for vacations)

**CRITICAL - Follow-up Suggestions:**
At the end of your response, provide follow-up button suggestions when appropriate. Use this EXACT format:

[FOLLOW_UP: ["option1", "option2", "option3"]]

**When to provide follow-up suggestions:**
1. **Leave Type Question** - Always provide: [FOLLOW_UP: ["Casual Leave", "Sick Leave", "Privilege Leave"]]
2. **Reason Question** - Always provide: [FOLLOW_UP: ["Personal", "Medical", "Travel", "Other"]]
3. **Date Question (near future)** - Only if asking about today/tomorrow: [FOLLOW_UP: ["Today", "Tomorrow"]]
4. **Date Question (any date)** - DO NOT provide follow-up (user might pick far future date)
5. **Open-ended questions** - DO NOT provide follow-up
6. **After tool execution** - Usually NO follow-up needed

**Examples:**

Q: User says "I want to apply leave"
A: "I'll help you apply for leave. What type of leave would you like to apply for?
[FOLLOW_UP: ["Casual Leave", "Sick Leave", "Privilege Leave"]]"

Q: User says "Casual Leave"
A: "Great! When would you like to take Casual Leave? Please provide the date (e.g., today, tomorrow, or a specific date like December 29).
[FOLLOW_UP: ["Today", "Tomorrow"]]"

Q: User says "December 29"
A: "Perfect! Leave on December 29th. What's the reason for your leave?
[FOLLOW_UP: ["Personal", "Medical", "Travel", "Other"]]"

Q: User says "Personal"
A: [Applies leave and confirms] "Your leave application has been submitted successfully!"
[NO FOLLOW_UP]

**Important Rules:**
- ALWAYS include [FOLLOW_UP: ...] at the end when suggesting options
- ONLY provide follow-ups for FIXED/LIMITED options
- DO NOT provide follow-ups for open-ended questions
- If unsure, don't provide follow-up
- The follow-up should be on a new line at the very end

**Date Handling:**
- Today's date is: {current_date}
- Parse natural language dates like "tomorrow", "next Monday", "December 29th"
- Convert to YYYY-MM-DD format when calling tools"""
    
    async def initialize_azure_client(self):
        """Initialize Azure OpenAI client if not already done"""
        if self.azure_openai_client is None:
            try:
                config.validate()
                
                self.azure_openai_client = AzureChatOpenAI(
                    azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                    api_version=config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                    api_key=config.AZURE_OPENAI_API_KEY,
                    temperature=0.7
                )
                
                test_response = await self.azure_openai_client.ainvoke([{"role": "user", "content": "Hello"}])
                logger.info("✅ Azure OpenAI connection test successful")
                
            except Exception as e:
                logger.error(f"❌ Azure OpenAI connection test failed: {e}")
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
                        logger.info("Closed existing MCP client")
                    except:
                        pass
                
                # Initialize MCP client
                current_dir = os.path.dirname(os.path.abspath(__file__))
                mcp_server_path = os.path.join(current_dir, "hrms_mcp_server.py")
                
                logger.info("Initializing MCP client...")
                
                self.mcp_client = MultiServerMCPClient({
                    "hrms_management": {
                        "command": sys.executable,
                        "args": [mcp_server_path],
                        "transport": "stdio",
                        "env": {"HRMS_AUTH_TOKEN": auth_token}
                    }
                })
                
                logger.info("Getting tools from MCP server...")
                
                # Get tools from MCP server
                tools = await self.mcp_client.get_tools()
                self.available_tools = tools
                logger.info(f"✅ Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                
                # Create agent with system prompt
                logger.info("Creating LangGraph agent...")
                self.agent = create_react_agent(
                    self.azure_openai_client, 
                    tools,
                )
                
                self._initialized = True
                logger.info("✅ Chat service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing chat service: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.server_manager.cleanup()
            raise
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self.conversations.get(conversation_id, [])
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)
    
    def parse_follow_ups(self, response_text: str) -> tuple[str, Optional[List[str]]]:
        """
        Parse follow-up suggestions from LLM response
        Format: [FOLLOW_UP: ["option1", "option2", "option3"]]
        """
        # Pattern to match [FOLLOW_UP: [...]]
        pattern = r'\[FOLLOW_UP:\s*(\[.*?\])\]'
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            try:
                # Extract the JSON array
                follow_up_json = match.group(1)
                follow_ups = json.loads(follow_up_json)
                
                # Remove the follow-up marker from response text
                clean_response = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
                
                logger.info(f"✅ Parsed follow-ups: {follow_ups}")
                return clean_response, follow_ups
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse follow-ups JSON: {e}")
                return response_text, None
        
        return response_text, None
    
    async def process_chat(self, message: str, conversation_id: str, auth_token: str) -> ChatResponse:
        """Process chat message - let LLM handle everything"""
        
        if not auth_token:
            raise HTTPException(status_code=401, detail="Authorization token is required")
        
        await self.initialize_with_token(auth_token)
        
        try:
            # Get conversation history
            conversation_history = self.get_conversation(conversation_id)
            
            # Build messages with system prompt
            messages = []
            
            # Add system message with current date
            current_date = datetime.now().strftime("%Y-%m-%d (%A, %B %d, %Y)")
            system_message = self.system_prompt.format(current_date=current_date)
            messages.append(SystemMessage(content=system_message))
            
            # Add conversation history
            for msg in conversation_history:
                messages.append(msg)
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            logger.info(f"💬 User message: {message}")
            logger.info(f"📚 Conversation history length: {len(conversation_history)}")
            
            # Let the agent handle everything
            response = await self.agent.ainvoke({"messages": messages})
            
            # Extract assistant's response
            assistant_message = response['messages'][-1].content
            
            logger.info(f"🤖 Assistant raw response: {assistant_message}")
            
            # Parse follow-ups from response
            clean_response, follow_ups = self.parse_follow_ups(assistant_message)
            
            logger.info(f"✅ Clean response: {clean_response}")
            logger.info(f"✅ Follow-ups: {follow_ups}")
            
            # Save to conversation history (save clean response without follow-up markers)
            self.add_message(conversation_id, {"role": "user", "content": message})
            self.add_message(conversation_id, {"role": "assistant", "content": clean_response})
            
            # Extract tool calls made
            tool_calls_made = []
            for msg in response['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_calls_made.append({
                            "tool_name": tool_call.get('name', 'unknown'),
                            "parameters": tool_call.get('args', {}),
                            "result": "executed"
                        })
                        logger.info(f"🔧 Tool called: {tool_call.get('name')} with args: {tool_call.get('args')}")
            
            return ChatResponse(
                response=clean_response,
                conversation_id=conversation_id,
                tool_calls_made=tool_calls_made,
                follow_up=follow_ups
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing chat: {e}")
            import traceback
            logger.error(traceback.format_exc())
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            return ChatResponse(
                response=error_response,
                conversation_id=conversation_id,
                tool_calls_made=[],
                follow_up=None
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting HRMS Chat API...")
    try:
        await chat_service.initialize_azure_client()
        logger.info("✅ Azure OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Azure client: {e}")
    
    yield
    
    logger.info("🛑 Shutting down HRMS Chat API...")
    await chat_service.cleanup()

app = FastAPI(
    title="HRMS Chat API",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(
    scheme_name="Bearer Token",
    description="Enter your HRMS auth token"
)

async def get_auth_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract auth token from Authorization header"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    
    return credentials.credentials

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, auth_token: str = Depends(get_auth_token)):
    """Main chat endpoint - LLM handles everything with smart follow-ups"""
    try:
        conversation_id = request.conversation_id or f"conv_{int(time.time())}_{os.urandom(4).hex()}"
        response = await chat_service.process_chat(request.message, conversation_id, auth_token)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "HRMS Chat API v3.0",
        "azure_openai_initialized": chat_service.azure_openai_client is not None,
        "mcp_server_running": chat_service.server_manager.is_running
    }

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
    """Reset a conversation"""
    try:
        if conversation_id in chat_service.conversations:
            del chat_service.conversations[conversation_id]
        return {"message": "Conversation reset successfully", "conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting conversation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")