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
        self.mcp_client = None
        self.agent = None
        self._initialized = False
        self.server_manager = MCPServerManager()
        self.azure_openai_client = None
    
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
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)
    
    async def process_chat(self, message: str, conversation_id: str, auth_token: str) -> ChatResponse:
        """Process chat message with auth token"""
        
        # Ensure initialization with current auth token
        if not auth_token:
            raise HTTPException(status_code=401, detail="Authorization token is required")
        
        await self.initialize_with_token(auth_token)
        
        try:
            # Get conversation history
            conversation_history = self.get_conversation(conversation_id)
            
            # Build messages for the agent
            messages = []
            for msg in conversation_history:
                messages.append(msg)
            messages.append({"role": "user", "content": message})
            
            # Process with LangGraph agent
            response = await self.agent.ainvoke({"messages": messages})
            
            # Extract the assistant's response
            assistant_message = response['messages'][-1].content
            
            # Update conversation history
            self.add_message(conversation_id, {"role": "user", "content": message})
            self.add_message(conversation_id, {"role": "assistant", "content": assistant_message})
            
            # Extract tool calls information
            tool_calls_made = []
            for msg in response['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_calls_made.append({
                            "tool_name": tool_call.get('name', 'unknown'),
                            "parameters": tool_call.get('args', {}),
                            "result": "executed"
                        })
            
            return ChatResponse(
                response=assistant_message,
                conversation_id=conversation_id,
                tool_calls_made=tool_calls_made
            )
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            return ChatResponse(
                response=error_response,
                conversation_id=conversation_id,
                tool_calls_made=[]
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
    """Main chat endpoint with dynamic auth token"""
    try:
        conversation_id = request.conversation_id or f"conv_{len(chat_service.conversations) + 1}"
        response = await chat_service.process_chat(request.message, conversation_id, auth_token)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "HRMS Chat API v1.0",
        "azure_openai_initialized": chat_service.azure_openai_client is not None,
        "mcp_server_running": chat_service.server_manager.is_running,
        "azure_openai_endpoint": config.AZURE_OPENAI_ENDPOINT,
        "azure_openai_deployment": config.AZURE_OPENAI_DEPLOYMENT_NAME,
        "auth_token_required": True
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check configuration"""
    return {
        "environment_variables": {
            "AZURE_OPENAI_API_KEY": "***" if config.AZURE_OPENAI_API_KEY else "NOT_SET",
            "AZURE_OPENAI_ENDPOINT": config.AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_VERSION": config.AZURE_OPENAI_API_VERSION,
            "AZURE_OPENAI_DEPLOYMENT_NAME": config.AZURE_OPENAI_DEPLOYMENT_NAME,
        },
        "working_directory": os.getcwd(),
        "files_present": os.listdir(os.getcwd()),
        "mcp_server_running": chat_service.server_manager.is_running,
        "current_auth_token_set": chat_service.server_manager.current_auth_token is not None
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "HRMS Chat API",
        "version": "1.0.0",
        "status": "running",
        "auth_required": True,
        "endpoints": {
            "chat": "POST /chat (requires Authorization header)",
            "health": "GET /health",
            "debug": "GET /debug",
            "tools": "GET /tools (requires Authorization header)"
        },
        "usage": {
            "auth_header": "Authorization: Bearer <your-token>",
            "note": "Use your HRMS authentication token"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)), log_level="info")