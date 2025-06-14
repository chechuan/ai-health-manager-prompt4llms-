import asyncio
import json
import threading
from contextlib import AsyncExitStack
from typing import Dict, Optional, Union

from dotenv import load_dotenv

from src.pkgs.models.proactivate_service.qwen_agent.log import logger
from src.pkgs.models.proactivate_service.qwen_agent.tools.base import BaseTool, register_tool


class MCPManager:
    _instance = None  # Private class variable to store the unique instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCPManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'clients'):
            """Set a new event loop in a separate thread"""
            load_dotenv()  # Load environment variables from .env file
            self.clients: dict = {}
            self.exit_stack = AsyncExitStack()
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self.start_loop, daemon=True)
            self.loop_thread.start()

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def is_valid_mcp_servers(self, config: dict):
        """Example of mcp servers configuration:
        {
         "mcpServers": {
            "memory": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"]
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
            }
         }
        }
        """

        # Check if the top-level key "mcpServers" exists and its value is a dictionary
        if not isinstance(config, dict) or 'mcpServers' not in config or not isinstance(config['mcpServers'], dict):
            return False
        mcp_servers = config['mcpServers']
        # Check each sub-item under "mcpServers"
        for key in mcp_servers:
            server = mcp_servers[key]
            # Each sub-item must be a dictionary
            if not isinstance(server, dict):
                return False
            if 'command' in server:
                # "command" must be a string
                if not isinstance(server['command'], str):
                    return False
                # "args" must be a list
                if 'args' not in server or not isinstance(server['args'], list):
                    return False
            if 'url' in server:
                # "url" must be a string
                if not isinstance(server['url'], str):
                    return False
                # "headers" must be a dictionary
                if 'headers' in server and not isinstance(server['headers'], dict):
                    return False
            # If the "env" key exists, it must be a dictionary
            if 'env' in server and not isinstance(server['env'], dict):
                return False
        return True

    def initConfig(self, config: Dict):
        logger.info(f'Initialize from config {config}. ')
        if not self.is_valid_mcp_servers(config):
            raise ValueError('Config format error')
        # Submit coroutine to the event loop and wait for the result
        future = asyncio.run_coroutine_threadsafe(self.init_config_async(config), self.loop)
        try:
            result = future.result()  # You can specify a timeout if desired
            return result
        except Exception as e:
            logger.info(f'Error executing function: {e}')
            return None

    async def init_config_async(self, config: Dict):
        tools: list = []
        mcp_servers = config['mcpServers']
        for server_name in mcp_servers:
            client = MCPClient()
            server = mcp_servers[server_name]
            await client.connection_server(self.exit_stack, server)  # Attempt to connect to the server
            self.clients[server_name] = client  # Add to clients dict after successful connection
            for tool in client.tools:
                """MCP tool example:
                {
                "name": "read_query",
                "description": "Execute a SELECT query on the SQLite database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                        "type": "string",
                        "description": "SELECT SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
                """
                parameters = tool.inputSchema
                # The required field in inputSchema may be empty and needs to be initialized.
                if 'required' not in parameters:
                    parameters['required'] = []
                # Remove keys from parameters that do not conform to the standard OpenAI schema
                # Check if the required fields exist
                required_fields = {'type', 'properties', 'required'}
                missing_fields = required_fields - parameters.keys()
                if missing_fields:
                    raise ValueError(f'Missing required fields in schema: {missing_fields}')

                # Keep only the necessary fields
                cleaned_parameters = {
                    'type': parameters['type'],
                    'properties': parameters['properties'],
                    'required': parameters['required']
                }
                register_name = server_name + '-' + tool.name
                agent_tool = self.create_tool_class(register_name, server_name, tool.name, tool.description,
                                                    cleaned_parameters)
                tools.append(agent_tool)
        return tools

    def create_tool_class(self, register_name, server_name, tool_name, tool_desc, tool_parameters):

        @register_tool(register_name)
        class ToolClass(BaseTool):
            description = tool_desc
            parameters = tool_parameters

            def call(self, params: Union[str, dict], **kwargs) -> str:
                tool_args = json.loads(params)
                # Submit coroutine to the event loop and wait for the result
                manager = MCPManager()
                client = manager.clients[server_name]
                future = asyncio.run_coroutine_threadsafe(client.execute_function(tool_name, tool_args), manager.loop)
                try:
                    result = future.result()
                    return result
                except Exception as e:
                    logger.info(f'Error executing function: {e}')
                    return None
                return 'Function executed'

        ToolClass.__name__ = f'{register_name}_Class'
        return ToolClass()

    async def clearup(self):
        await self.exit_stack.aclose()


class MCPClient:

    def __init__(self):
        from mcp import ClientSession

        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.tools: list = None

    async def connection_server(self, exit_stack, mcp_server):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.sse import sse_client
        """Connect to an MCP server and retrieve the available tools."""
        try:
            if 'url' in mcp_server:
                self._streams_context = sse_client(url=mcp_server.get('url'), headers=mcp_server.get('headers', {"Accept": "text/event-stream"}))
                streams = await self._streams_context.__aenter__()

                self._session_context = ClientSession(*streams)
                self.session: ClientSession = await self._session_context.__aenter__()
            else:
                server_params = StdioServerParameters(
                    command = mcp_server["command"],
                    args = mcp_server["args"],
                    env = mcp_server.get("env", None)
                )
                stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                self.stdio, self.write = stdio_transport
                self.session = await exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()

            list_tools = await self.session.list_tools()
            self.tools = list_tools.tools
        except Exception as e:
            logger.info(f"Failed to connect to server: {e}")

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def execute_function(self, tool_name, tool_args: dict):
        response = await self.session.call_tool(tool_name, tool_args)
        texts = []
        for content in response.content:
            if content.type == 'text':
                texts.append(content.text)
        if texts:
            return '\n\n'.join(texts)
        else:
            return 'execute error'
