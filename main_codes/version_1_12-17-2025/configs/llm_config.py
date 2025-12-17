
from http.client import HTTPException
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
import httpx
import asyncio

# Load environment variables
load_dotenv()

# Nexus configuration
BASE_URL = os.getenv("NEXUS_OS_BASE_ENDPOINT")
AUTH_URL = os.getenv("NEXUS_OS_AUTH_ENDPOINT")
BASIC_AUTH= os.getenv("NEXUS_OS_BASICAUTH_BASE64")
CHAT_ENDPOINT = os.getenv("NEXUS_OS_CHAT_COMPLETIONS")
MAX_TOKENS = int(os.getenv("NEXUS_OS_MAX_TOKENS", "2000"))

base_endpoint = BASE_URL
chat_completions_path = CHAT_ENDPOINT
auth_endpoint = AUTH_URL
basic_auth_base64 = BASIC_AUTH

url = f"{base_endpoint}{chat_completions_path}"

async def get_bearer_token() -> str:
        """
        Step 1: Get bearer token using OAuth2 client credentials flow
        """
        headers = {
            'Authorization': f'Basic {basic_auth_base64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'scope': 'openid'
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    auth_endpoint,
                    headers=headers,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                
                token_data = response.json()
                access_token = token_data.get('access_token')
                
                if not access_token:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to obtain access token from Nexus OS"
                    )
                
                # self.logger.debug("Successfully obtained bearer token from Nexus OS")
                return access_token
                
        except httpx.HTTPError as e:
            # self.logger.error(f"HTTP error during token request: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to authenticate with Nexus OS: {str(e)}"
            )
        except Exception as e:
            # self.logger.error(f"Unexpected error during token request: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Authentication error: {str(e)}"
            )


def prepare_chat_request(messages):
        """
        Prepare chat completion request in Nexus OS format
        """
        # messages = [
        #     {"role": "user", "content": "What is the capital of France?"}
        # ]
        prompt =messages
        sessionid="test_session_123"
        client_app="VBCG_TEST"
        temperature=0.7
        # Determine role and prompt format based on message structure
        if isinstance(messages, str):
            # Simple string prompt
            role = "user"
            prompt = messages
        elif isinstance(messages, list):
            # Check if it's a single message or multiple messages
            if len(messages) == 1 and isinstance(messages[0], dict):
                # Single message - extract content as string prompt
                role = "user"
                prompt = messages[0].get('content', str(messages[0]))
            else:
                # Multiple messages - send as array and determine role
                roles_present = set()
                for msg in messages:
                    if isinstance(msg, dict) and 'role' in msg:
                        roles_present.add(msg['role'])
                
                # Set role based on roles present
                if len(roles_present) > 1 or 'system' in roles_present or 'assistant' in roles_present:
                    role = "both"
                else:
                    role = "user"
                prompt = messages
        else:
            # Fallback
            role = "user"
            prompt = messages
        
        return {
            "prompt": prompt,
            "user_id": sessionid,
            "app_id":"APP-7301",
            "agent_id":"agent_rom_estimator",
            "session_id": sessionid,
            "event": client_app,
            "host_system":"agent_risant_aisvc",
            "user_name": client_app,
            "ip_address": "127.0.0.1",
            "model": "gpt-4o",
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
            "role": "user"

        }

async def chat_completions(messages):
    
    bearer_token = await get_bearer_token()
    headers = {
                'Authorization': f'Bearer {bearer_token}',
                'Content-Type': 'application/json'
            }
    
    request_data= prepare_chat_request(messages)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers=headers,
                json=request_data,
                timeout=60.0
            )
            response.raise_for_status()
            print('respond from nexus')
            nexus_response = response.json()
            return nexus_response
            # # Extract the final_result from Nexus OS response
            # final_result = nexus_response.get('final_result', '')
            # print(final_result)
        except httpx.HTTPStatusError as e:
            print(e.response.text)
            return None

if __name__ == '__main__' :
     
    result=asyncio.run(chat_completions())
    print(result)