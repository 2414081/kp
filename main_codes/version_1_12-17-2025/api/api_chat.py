import requests
import re

def send_question_to_chat_api(question):
    """
    Sends a question to the Care Guide Chat API and returns:
      - original question
      - summarization (falls back to joined snippets if missing)
      - list of citation snippets
    """
    url = "https://api-az-gtwy-dev.kp.org/api/vbp/ai/data/v1/api/careguidechat?envlbl=DEV2"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Basic U1ZDX0FQUC03MzAxXzFfVFM5MDc2X05QOnN3JXB0XEFDcDZ+MzJDR1Q=",
        "x-correlationid": "test123"
    }
    payload = {
        "metadata": {
            "uId": "4t128764182648",
            "source": "Geisinger-EPIC",
            "appId": "VBCG",
            "affiliate": "geisinger",
            "subAffiliate": "Epic",
            "product": "Value Based Care Guide",
            "language": "en"
        },
        "interaction": [
            {
                "order": 1,
                "type": "chat",
                "query": question
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 500:
            return {"error": "Server error occurred. Please try again later."}

        try:
            api_response = response.json()
        except ValueError:
            return {"error": "Invalid JSON response from the server."}

        if "interaction" in api_response and api_response["interaction"]:
            interaction_obj = api_response["interaction"][0]
            citations = interaction_obj.get("citations", [])
            snippets = [c.get("snippet", "") for c in citations if c.get("snippet")]

            # Get summarization if present; fallback to snippets when missing/empty
            raw_summary = interaction_obj.get("summarization")  # No default so we can detect absence
            if raw_summary is None or not str(raw_summary).strip() or str(raw_summary).strip() == "No summarization found":
                if snippets:
                    # Use joined snippets as summary fallback
                    clean_summary = re.sub(r'\s+', ' ', "; ".join(snippets)).strip()
                else:
                    clean_summary = "No summarization available"
            else:
                clean_summary = re.sub(r'#', '', raw_summary)
                clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
                # If after cleaning it's empty, still fallback to snippets
                if not clean_summary and snippets:
                    clean_summary = re.sub(r'\s+', ' ', "; ".join(snippets)).strip()
                elif not clean_summary:
                    clean_summary = "No summarization available"

            # Clean each snippet individually
            cleaned_snippets = [re.sub(r'\s+', ' ', snippet).strip() for snippet in snippets]

            return {
                "question": interaction_obj.get("query", "No query found"),
                "summarization": clean_summary,
                "snippets": cleaned_snippets  # Now returns list of individual snippets
            }
        else:
            return {"error": "Unexpected response structure from the API."}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# # Example call
# response = send_question_to_chat_api("What is cardiology?")
# print(response)

