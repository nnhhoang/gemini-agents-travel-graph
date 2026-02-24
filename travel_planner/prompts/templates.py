"""
Prompt template management.

Templates are stored in DynamoDB with versioning and A/B test support.
"""

from pydantic import BaseModel, Field, computed_field


def render_template(template: str, **kwargs: str) -> str:
    """Render a template string, leaving unresolved vars as-is."""
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", value)
    return result


class PromptTemplate(BaseModel):
    """Prompt template with DynamoDB keys."""

    template_id: str
    version: int
    template: str
    status: str = "active"
    description: str | None = None
    ab_variant: str | None = None

    @computed_field
    @property
    def pk(self) -> str:
        return f"PROMPT#{self.template_id}"

    @computed_field
    @property
    def sk(self) -> str:
        return f"VERSION#{self.version:03d}"

    @computed_field
    @property
    def gsi1pk(self) -> str:
        return f"PROMPT#{self.template_id}#ACTIVE"

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def render(self, **kwargs: str) -> str:
        """Render this template with the given variables."""
        return render_template(self.template, **kwargs)
