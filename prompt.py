class Prompt():
    def __init__(self) -> None:
        self.system_prompt = \
"""You are tasked to predict the labels from a tabular dataset. 
Please output ONLY your predicted label key. DO NOT OUTPUT ANYTHING ELSE!"""

        self.user_prompt = \
"""Here are some Dataset examples:
{D}
Given the Dataset examples, predict the "{label_name}" of the following. Please output ONLY your predicted {label_name} label key from {label_keys}. DO NOT OUTPUT ANYTHING ELSE!:
{note}
"{label_name}" takes the form of the following: {labels}.
Let me repeat again, output your predicted {label_name} label key from {label_keys} ONLY. DON'T OUTPUT ANYTHING ELSE!"""

    def get_system_prompt(self):
        return self.system_prompt
    
    def get_user_prompt(self, dataset, label_name, label_keys, labels, note):
        return self.user_prompt.format(D=dataset, label_name=label_name, label_keys=label_keys, labels=labels, note=note)