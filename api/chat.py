
import akshare as ak
import json
import requests
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3-flash-preview"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

tools_def = [
    {
        "name": "get_stock_price",
        "description": "获取A股股票实时行情，包括价格、涨跌幅等",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "6位股票代码，例如000001"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_stock_history",
        "description": "获取股票历史K线数据",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "6位股票代码"},
                "start_date": {"type": "string", "description": "开始日期，格式YYYYMMDD"},
                "end_date": {"type": "string", "description": "结束日期，格式YYYYMMDD"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_index_spot",
        "description": "获取沪深指数实时行情，包括上证、深证、创业板等",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

def call_akshare_tool(name, args):
    try:
        if name == "get_stock_price":
            df = ak.stock_zh_a_spot_em()
            result = df[df['代码'] == args.get('symbol', '')].to_dict('records')
            return {"data": result}
        elif name == "get_stock_history":
            df = ak.stock_zh_a_hist(
                symbol=args['symbol'],
                period='daily',
                start_date=args.get('start_date', '20240101'),
                end_date=args.get('end_date', '20251231'),
                adjust="qfq"
            )
            return {"data": df.tail(10).to_dict('records')}
        elif name == "get_index_spot":
            df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
            return {"data": df.to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

def send_to_gemini(messages):
    payload = {
        "contents": messages,
        "tools": [{"function_declarations": tools_def}]
    }
    resp = requests.post(API_URL, json=payload, timeout=30)
    return resp.json()

@app.route("/api/chat", methods=["POST"])
def chat():
    question = request.json.get("question", "")
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    messages = [{"role": "user", "parts": [{"text": question}]}]

    for _ in range(5):
        result = send_to_gemini(messages)
        if "error" in result:
            return jsonify({"error": str(result["error"])}), 500

        parts = result['candidates'][0]['content'].get('parts', [])
        func_calls = [p for p in parts if 'functionCall' in p]
        text_parts = [p for p in parts if 'text' in p]

        if func_calls:
            messages.append({"role": "model", "parts": parts})
            tool_results = []
            for fc in func_calls:
                fn_name = fc['functionCall']['name']
                fn_args = fc['functionCall'].get('args', {})
                fn_result = call_akshare_tool(fn_name, fn_args)
                tool_results.append({
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": json.dumps(fn_result, ensure_ascii=False, default=str)}
                    }
                })
            messages.append({"role": "user", "parts": tool_results})
        elif text_parts:
            answer = "\n".join(p['text'] for p in text_parts)
            return jsonify({"answer": answer})
        else:
            break

    return jsonify({"error": "未能获取回答"}), 500

if __name__ == "__main__":
    app.run(debug=True)
