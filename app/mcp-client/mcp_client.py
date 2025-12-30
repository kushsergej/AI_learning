# from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# from dotenv import load_dotenv
from pathlib import Path



class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    def connect_to_server(self, server_script_path: str):
        '''Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script
        '''
        path = Path(server_script_path).resolve()
        server_params = StdioServerParameters(
            command="python3",
            args=[path.name],
            env=None,
        )
        stdio_transport = self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        self.session.initialize()
        
        # List available tools
        response = self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        self.exit_stack.aclose()



def main():
    client = MCPClient()
    try:
        client.connect_to_server(server_script_path='app/mcp-server/mcp_server.py')
        # client.chat_loop()
    finally:
        client.cleanup()



if __name__ == '__main__':
    main()