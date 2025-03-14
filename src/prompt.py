
PROMPT_TYPES_TO_TEXT_TEMPLATE = {
    "tabular": \
"""{icl}
{note} <output>""",
    "regression": \
"""{icl}
 {note} <output>"""
}

class Prompt():
    def __init__(self, label_name, label_map, label_keys, prompt_type = "tabular") -> None:
        self.label_name = label_name
        self.label_map = label_map
        self.label_keys = label_keys
        self.prompt_type = prompt_type

    @property
    def prompt_text(self) -> str:
        return PROMPT_TYPES_TO_TEXT_TEMPLATE[self.prompt_type]
            
    def get_puzD_prompt(self, z, D):
        return self.prompt_text.format(self=self, note=z, icl=D)

    def get_pyxuzD_prompt(self, x, icl):
        return self.prompt_text.format(self=self, note=x, icl=icl)

    def get_pyxD_prompt(self, x, D):
        return self.prompt_text.format(self=self, note=x, icl=D)
    
class ToyClassificationPrompt(Prompt):
    def __init__(self, label_name, label_map, label_keys) -> None:
        super().__init__(label_name, label_map, label_keys, prompt_type="toy_classification")      
