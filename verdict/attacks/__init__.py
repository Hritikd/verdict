"""Attack implementations for Verdict red-teaming."""
from verdict.attacks.pair import PAIRAttack, PAIRConfig
from verdict.attacks.templates import TemplateAttack, TemplateConfig, ALL_TEMPLATES
from verdict.attacks.crescendo import CrescendoAttack, CrescendoConfig
from verdict.attacks.injection import PromptInjectionAttack, InjectionConfig

__all__ = [
    "PAIRAttack", "PAIRConfig",
    "TemplateAttack", "TemplateConfig", "ALL_TEMPLATES",
    "CrescendoAttack", "CrescendoConfig",
    "PromptInjectionAttack", "InjectionConfig",
]
