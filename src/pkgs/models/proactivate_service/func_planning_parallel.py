import json
import os

from src.pkgs.models.proactivate_service.qwen_agent.llm import get_chat_model


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def modify_schedule():
    """Get the current weather in a given location"""
    pass

def generate_glucose_alert_plan():
    """Get the current weather in a given location"""
    pass

def modify_health_promotion_plan():
    """Get the current weather in a given location"""
    pass


def test():
    llm = get_chat_model({
        'model': 'Qwen2.5-7B-Instruct',
        'model_server': 'http://10.228.67.99:26928/v1',
        'api_key': 'sk-a3vqxmzc4qvmJo6M360686EeA9Cc4dC4B2F65c000d60B8A5',
        'generate_cfg': {
            'fncall_prompt_type': 'qwen'
        }
    })

    # Step 1: send the conversation and available functions to the model
    messages = [{
        'role': 'user',
        'content': "用户空腹血糖值7.9",
    }]
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
        # {
        #     'name': 'generate_glucose_alert_plan',
        #     'description': '',
        #     'parameters': {
        #         'type': 'object',
        #         'properties': {},
        #         'required': [],
        #     },
        # },
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
            extra_generate_cfg=dict(
                # This will truncate the history until the input tokens are less than the limit.
                max_input_tokens=6500,

                # Note: set parallel_function_calls=True to enable parallel function calling
                parallel_function_calls=True,  # Default: False
                # Note: set function_choice='auto' to let the model decide whether to call a function or not
                # function_choice='auto',  # 'auto' is the default if function_choice is not set
                # Note: set function_choice='get_current_weather' to force the model to call this function
                # function_choice='get_current_weather',
            ),
    ):
        print(responses)

    messages.extend(responses)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    if isinstance(responses, list):
        fncall_msgs = [rsp for rsp in responses if rsp.get('function_call', None)]
    elif responses.get('function_call', None):
        fncall_msgs = [responses]
    else:
        fncall_msgs = []
    if fncall_msgs:
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            'modify_sports_schedule': modify_sports_schedule, 'generate_glucose_alert_plan': generate_glucose_alert_plan,
            'modify_health_promotion_plan': modify_health_promotion_plan
        }  # only one function in this example, but you can have multiple

        for msg in fncall_msgs:
            # Step 3: call the function
            print('# Function Call:')
            function_name = msg['function_call']['name']
            function_to_call = available_functions[function_name]
            function_args = json.loads(msg['function_call']['arguments'])
            function_response = function_to_call()
            print('# Function Response:')
            print(function_response)
            # Step 4: send the info for each function call and function response to the model
            # Note: please put the function results in the same order as the function calls
            messages.append({
                'role': 'function',
                'name': function_name,
                'content': function_response,
            })  # extend conversation with function response

        print('# Assistant Response 2:')
        for responses in llm.chat(
                messages=messages,
                functions=functions,
                extra_generate_cfg={
                    'max_input_tokens': 6500,
                    'parallel_function_calls': True,
                },
                stream=False,
        ):  # get a new response from the model where it can see the function response
            print(responses)


if __name__ == '__main__':
    test()