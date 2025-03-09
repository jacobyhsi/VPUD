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

# class PromptwText():
#     def __init__(self, label_name, label_map, label_keys) -> None:
#         self.label_name = label_name
#         self.label_map = label_map
#         self.label_keys = label_keys

#         self.puzD_prompt = \
# """The following are in-context example(s):

# {D}

# Given the example(s) as context, predict the "{self.label_name}" of the following sample:

# - {z}

# Label "{self.label_name}" takes the form of the following: {self.label_map}.

# Please output ONLY the "{self.label_name}" label key as your prediction.
# Enclose your output within a dictionary in <output> </output> tags i.e. <output> label_key </output>.
# **DO NOT OUTPUT ANYTHING ELSE!**
# """

#         self.puzD_prompt1 = \
# """The following are in-context example(s):

# {D}

# Given the example(s) as context, predict the "{self.label_name}" of the following sample:

# - {z}

# Label "{self.label_name}" takes the form of the following: {self.label_map}.

# Please output ONLY the "{self.label_name}" label key as your prediction.
# Enclose your output within a dictionary in <output> </output> tags i.e. <output> label_key </output>.
# **DO NOT OUTPUT ANYTHING ELSE!**"""

#         self.pyxuzD_prompt = \
# """The following are in-context example(s):

# {D}

# Followed by the another example:

# - {z} -> {self.label_name}: {label}

# Given the example(s) as context, predict the "{self.label_name}" of the following sample:

# - {x}

# Label "{self.label_name}" takes the form of the following: {self.label_map}.

# Please output ONLY the "{self.label_name}" label key.
# Enclose your output within a dictionary in <output> </output> tags i.e. <output> label_key </output>.
# **DO NOT OUTPUT ANYTHING ELSE!**"""

#     def get_puzD_prompt(self, z, D):
#         return self.puzD_prompt.format(self=self, z=z, D=D)

#     def get_pyxuzD_prompt(self, x, z, D, label):
#         return self.pyxuzD_prompt.format(self=self, x=x, z=z, D=D, label=label)

# class PromptProba():
#     def __init__(self, label_name, label_map, label_keys) -> None:
#         self.label_name = label_name
#         self.label_map = label_map
#         self.label_keys = label_keys

#         self.puzD_prompt = \
# """The following are in-context example(s):

# {D}

# Given the example(s) as context, predict the "{self.label_name}" of the following sample:

# - {z}

# Label "{self.label_name}" takes the form of the following: {self.label_map}.

# Please output ONLY the "{self.label_name}" label key of the highest predicted probability, along with its corresponding probability.
# Enclose your output within a dictionary in <output> </output> tags i.e. <output> {{label_key:probability}} </output>.
# Check and make sure that the total probabilities should add up to 1.0.
# **DO NOT OUTPUT ANYTHING ELSE!**"""

#         self.pyxuzD_prompt = \
# """The following are in-context example(s):

# {D}

# Followed by the another example:

# - {z} -> {self.label_name}: {label}

# Given the example(s) as context, predict the "{self.label_name}" of the following sample:

# - {x}

# Label "{self.label_name}" takes the form of the following: {self.label_map}.

# Please output ONLY the "{self.label_name}" label key of the highest predicted probability, along with its corresponding probability.
# Enclose your output within a dictionary in <output> </output> tags i.e. <output> {{label_key:probability}} </output>.
# Check and make sure that the total probabilities should add up to 1.0.
# **DO NOT OUTPUT ANYTHING ELSE!**"""

#     def get_puzD_prompt(self, z, D):
#         return self.puzD_prompt.format(self=self, z=z, D=D)

#     def get_pyxuzD_prompt(self, x, z, D, label):
#         return self.pyxuzD_prompt.format(self=self, x=x, z=z, D=D, label=label)
