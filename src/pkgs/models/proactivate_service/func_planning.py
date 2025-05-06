import json
import os

from qwen_agent.llm import get_chat_model

def test(fncall_prompt_type: str = 'qwen'):
    llm = get_chat_model({
        # Use the model service provided by DashScope:
        'model': 'Qwen2.5-7B-Instruct',
        'model_server': 'http://10.228.67.99:26928/v1',
        'api_key': 'sk-a3vqxmzc4qvmJo6M360686EeA9Cc4dC4B2F65c000d60B8A5',
        'generate_cfg': {
            'fncall_prompt_type': fncall_prompt_type
        },

        # Use the OpenAI-compatible model service provided by DashScope:
        # 'model': 'qwen2.5-72b-instruct',
        # 'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

        # Use the model service provided by Together.AI:
        # 'model': 'Qwen/qwen2.5-7b-instruct',
        # 'model_server': 'https://api.together.xyz',  # api_base
        # 'api_key': os.getenv('TOGETHER_API_KEY'),

        # Use your own model service compatible with OpenAI API:
        # 'model': 'Qwen/qwen2.5-7b-instruct',
        # 'model_server': 'http://localhost:8000/v1',  # api_base
        # 'api_key': 'EMPTY',
    })

    # Step 1: send the conversation and available functions to the model
    messages = [{'role': 'user', 'content': "早上空腹血糖值为7.5"}]
    functions = [
        {
            'name': 'modify_sports_schedule',
            'description': '',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
        {
            'name': 'generate_glucose_alert_plan',
            'description': '',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
        {
            'name': 'modify_health_promotion_plan',
            'description': '',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        }
    ]

    print('# Assistant Response 1:')
    responses = []
    for responses in llm.chat(
            messages=messages,
            functions=functions,
            stream=False,
            # Note: extra_generate_cfg is optional
            # extra_generate_cfg=dict(
            #     # Note: if function_choice='auto', let the model decide whether to call a function or not
            #     # function_choice='auto',  # 'auto' is the default if function_choice is not set
            #     # Note: set function_choice='get_current_weather' to force the model to call this function
            #     function_choice='get_current_weather',
            # ),
    ):
        print(responses)

    # If you do not need streaming output, you can either use the following trick:
    #   *_, responses = llm.chat(messages=messages, functions=functions, stream=True)
    # or use stream=False:
    #   responses = llm.chat(messages=messages, functions=functions, stream=False)

    messages.extend(responses)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    last_response = messages[-1]
    # if last_response.get('function_call', None):
    if messages.get('function_call', None):

        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            'get_current_weather': '',
        }  # only one function in this example, but you can have multiple
        function_name = last_response['function_call']['name']
        function_to_call = available_functions[function_name]
        function_args = json.loads(last_response['function_call']['arguments'])
        function_response = function_to_call(
            location=function_args.get('location'),
            unit=function_args.get('unit'),
        )
        print('# Function Response:')
        print(function_response)

        # Step 4: send the info for each function call and function response to the model
        messages.append({
            'role': 'function',
            'name': function_name,
            'content': function_response,
        })  # extend conversation with function response

        print('# Assistant Response 2:')
        for responses in llm.chat(
                messages=messages,
                functions=functions,
                stream=True,
        ):  # get a new response from the model where it can see the function response
            print(responses)


if __name__ == '__main__':
    # Run example of function calling with QwenFnCallPrompt
    test(fncall_prompt_type='qwen')

    # Run example of function calling with NousFnCallPrompt
    # test(fncall_prompt_type='nous')