def cls_intent(self, history, mid_vars, **kwargs):
    """意图识别
    """
    prefix = "Question" if history[-1]['role'] == "user" else "Answer"
    query = f"{prefix}: {history[-1]['content']}"

    # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
    if kwargs.get('intentPrompt', ''):
        prompt = kwargs.get('intentPrompt').format(h_p) + "\n\n" + query + "\nThought: "
    else:
        scene_prompt = get_parent_scene_intent(self.prompt_meta_data['intent'], kwargs.get('scene_code', 'default'))
        prompt = self.prompt_meta_data['intent']['意图模版']['description'].format(scene_prompt,
                                                                                   h_p) + "\n\n" + query + "\nThought: "

        # if kwargs.get('scene_code', 'default') == 'exhibition_hall_exercise':
        #     scene_prompt = get_scene_intent(self.prompt_meta_data['tool'], 'exhibition_hall_exercise')
        #     prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(scene_prompt, h_p) + "\n\n" + query + "\nThought: "
        # else:
        #     prompt = self.prompt_meta_data['tool']['父意图']['description'].format(h_p) + "\n\n" + query + "\nThought: "
    logger.debug('父意图模型输入：' + prompt)
    generate_text = callLLM(query=prompt, max_tokens=200, top_p=0.8,
                            temperature=0, do_sample=False, stop=['\nThought'], model='Qwen-14B-Chat')
    logger.debug('父意图识别模型输出：' + generate_text)
    intentIdx = 0
    if 'Intent:' in generate_text:
        intentIdx = generate_text.find("\nIntent: ") + 9
    elif '意图:' in generate_text:
        intentIdx = generate_text.find("\n意图:") + 4
    elif '\nFunction:' in generate_text:
        intentIdx = generate_text.find("\nFunction:") + 10
    text = generate_text[intentIdx:].split("\n")[0].strip()
    parant_intent = self.get_parent_intent_name(text)
    if parant_intent in ['呼叫五师', '音频播放', '生活工具查询', '医疗健康', '饮食营养', '运动咨询'] and (
            not kwargs.get('intentPrompt', '') or (
            kwargs.get('intentPrompt', '') and kwargs.get('subIntentPrompt', ''))):
        # sub_intent_prompt = self.prompt_meta_data['intent'][parant_intent]['description']
        if parant_intent in ['呼叫五师意图']:
            history = history[-1:]
            query = "\n".join(
                [("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
            h_p = '无'
        if kwargs.get('subIntentPrompt', ''):
            prompt = kwargs.get('subIntentPrompt').format(h_p) + "\n\n" + query + "\nThought: "
        else:
            scene_prompt = get_sub_scene_intent(self.prompt_meta_data['intent'], kwargs.get('scene_code', 'default'),
                                                parant_intent)
            prompt = self.prompt_meta_data['intent']['意图模版']['description'].format(scene_prompt,
                                                                                       h_p) + "\n\n" + query + "\nThought: "
            # prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(sub_intent_prompt, h_p) + "\n\n" + query + "\nThought: "
        logger.debug('子意图模型输入：' + prompt)
        generate_text = callLLM(query=prompt, max_tokens=200, top_p=0.8,
                                temperature=0, do_sample=False, stop=['\nThought'], model='Qwen-14B-Chat')
        logger.debug('子意图模型输出：' + generate_text)
        intentIdx = 0
        if 'Intent:' in generate_text:
            intentIdx = generate_text.find("\nIntent: ") + 9
        elif '意图:' in generate_text:
            intentIdx = generate_text.find("\n意图:") + 4
        elif '\nFunction:' in generate_text:
            intentIdx = generate_text.find("\nFunction:") + 10
        text = generate_text[intentIdx:].split("\n")[0]
    self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
    return text