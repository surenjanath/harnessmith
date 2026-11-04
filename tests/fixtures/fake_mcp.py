"""A minimal MCP stdio server for tests: one tool, `add`, that sums two numbers."""
import json, sys

for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue  # notification
    m, i = msg["method"], msg["id"]
    if m == "initialize":
        res = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "calc", "version": "1"}}
    elif m == "tools/list":
        res = {"tools": [{"name": "add", "description": "Add two numbers.", "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}]}
    elif m == "tools/call":
        a = msg["params"]["arguments"]
        res = {"content": [{"type": "text", "text": f"sum={a['a'] + a['b']}"}], "isError": False}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": i, "error": {"code": -32601, "message": "no such method"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": i, "result": res}), flush=True)
