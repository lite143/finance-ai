import akshare as ak
import json
import requests
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone, timedelta

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3-flash-preview"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

tools_def = [
    {
        "name": "get_stock_price",
        "description": "获取A股股票实时行情，包括当前价格、涨跌幅等",
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
        "description": "获取沪深指数今日实时行情，包括上证、深证、创业板、沪深300等当前最新数据",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

def call_akshare_tool(name, args):
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    try:
        if name == "get_stock_price":
            df = ak.stock_zh_a_spot_em()
            result = df[df['代码'] == args.get('symbol', '')].to_dict('records')
            return {"data": result, "query_time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")}

        elif name == "get_stock_history":
            end = args.get('end_date', today)
            start = args.get('start_date', '20250101')
            df = ak.stock_zh_a_hist(
                symbol=args['symbol'],
                period='daily',
                start_date=start,
                end_date=end,
                adjust="qfq"
            )
            return {"data": df.tail(10).to_dict('records'), "query_time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")}

        elif name == "get_index_spot":
            # 使用实时行情接口
            df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
            result = df.to_dict('records')
            return {"data": result, "query_time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")}

    except Exception as e:
        return {"error": str(e)}

def send_to_gemini(messages):
    today_str = datetime.now(timezone(timedelta(hours=8))).strftime("%Y年%m月%d日 %H:%M")
    system_instruction = f"你是一个金融AI助手。当前真实时间是{today_str}。请使用工具获取实时数据，回答时注明数据获取时间，不要编造数据。"
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
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
