# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-09 17:19:15
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''


def _parse_latest_plugin_call(text: str):
    h = text.find('Thought:')
    i = text.find('\nAction:')
    j = text.find('\nAction Input:')
    k = text.find('\nObservation:')
    l = text.find('\nFinal Answer:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_thought = text[h + len('Thought:'):i].strip()
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_thought, plugin_name, plugin_args
    elif l > 0:
        if h > 0:
            plugin_thought = text[h + len('Thought:'):l].strip()
            plugin_args = text[l + len('\nFinal Answer:'):].strip()
            plugin_args = plugin_args.split("\n")[0]
            return plugin_thought, "直接回复用户问题", plugin_args
        else:
            plugin_args = text[l + len('\nFinal Answer:'):].strip()
            return "I know the final answer.", "直接回复用户问题", plugin_args
    return '', ''


a = """
Thought: 请帮我呼叫张医生。
Final Answer: 我已经帮你呼叫张医生了，你的血压是180/120，建议你尽快去医院就诊。请保持心情愉悦，避免情绪波动，保持良好的生活习惯，定期监测血压，遵医嘱服药。祝你早日康复！
Thought: bingo
Final Answer: 好的，我已经帮你呼叫张医生了，你的血压是180/120，建议你尽快去医院就诊。请保持心情愉悦，避免情绪波动，保持良好的生活习惯，定期监测血压，遵医嘱服药。祝你早日康复！
Thought: bingo
Final Answer: 好的，我已经帮你呼叫张医生了，你的血压是180/120，建议你尽快去医院就诊。请保持心情愉悦，避免情绪波动，保持良好的生活习惯，定期监测血压，遵医嘱服药。祝你早日康复！
Thought: bingo
Final Answer"""
# text = _parse_latest_plugin_call(a)

