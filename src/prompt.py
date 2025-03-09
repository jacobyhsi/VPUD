class Prompt():
    def __init__(self, label_name, label_map, label_keys) -> None:
        self.label_name = label_name
        self.label_map = label_map
        self.label_keys = label_keys

        self.puzD_prompt = \
"""
{D}

Complete the following per the context provided above:
{z} <output>
"""

        self.pyxuzD_prompt = \
"""
{icl}

Complete the following per the context provided above:
{x} <output>
"""

        self.pyxD_prompt = \
"""
{D}

Complete the following per the context provided above:
{x} <output>
"""

    def get_puzD_prompt(self, z, D):
        return self.puzD_prompt.format(self=self, z=z, D=D)

    def get_pyxuzD_prompt(self, x, icl):
        return self.pyxuzD_prompt.format(self=self, x=x, icl=icl)

    def get_pyxD_prompt(self, x, D):
        return self.pyxD_prompt.format(self=self, x=x, D=D)
