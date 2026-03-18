import asyncio
import os
import json
from fastmcp import Client
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

## this client will call the local mcp server
# ── Hugging Face client ───────────────────────────────────────────────────────
hf_client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_API_KEY"],
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def build_system_prompt(tools: list) -> str:
    tool_descriptions = json.dumps(tools, indent=2)
    return f"""You are a helpful assistant with access to the following tools:

{tool_descriptions}

When you need to use a tool, respond ONLY with a JSON object in this exact format:
{{
  "tool": "<tool_name>",
  "input": {{ "<param>": <value> }}
}}

When you have a final answer (no more tools needed), respond normally in plain text.
"""


async def main():
    async with Client("server.py") as mcp_client:

        # ── 1. Fetch tools from MCP server ────────────────────────────────────
        mcp_tools = await mcp_client.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

        print("Tools available to LLM:")
        for t in tools:
            print(f"  • {t['name']}: {t['description']}")
        print()

        # ── 2. User message ───────────────────────────────────────────────────
        # user_message = "Please roll 4 dice and then add the numbers 8.5 and 1.5"
        user_message = input("You: ")
        print(f"User: {user_message}\n")

        messages = [
            {"role": "system", "content": build_system_prompt(tools)},
            {"role": "user",   "content": user_message},
        ]

        # ── 3. Agentic loop ───────────────────────────────────────────────────
        while True:
            response = hf_client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )

            reply = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": reply})

            # Try to parse a tool call from the LLM response
            try:
                clean = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                tool_call = json.loads(clean)

                if "tool" in tool_call and "input" in tool_call:
                    tool_name  = tool_call["tool"]
                    tool_input = tool_call["input"]

                    print(f"LLM calling tool: {tool_name}({tool_input})")
                    result = await mcp_client.call_tool(tool_name, tool_input)
                    print(f"  → Result: {result}\n")

                    messages.append({
                        "role": "user",
                        "content": f"Tool '{tool_name}' returned: {result}. Continue."
                    })
                    continue

            except (json.JSONDecodeError, KeyError):
                pass

            # Final answer
            print(f"Assistant: {reply}")
            break


if __name__ == "__main__":
    asyncio.run(main())
