"""Shared owner-only Discord view factories."""

from __future__ import annotations

from typing import Any


def build_common_views(discord: Any, owner_id: int) -> tuple[type[Any], type[Any]]:
    """Return owner-checked base and paginated view classes."""

    class OwnerView(discord.ui.View):
        async def interaction_check(self, interaction: Any) -> bool:
            if int(interaction.user.id) == owner_id:
                return True
            await interaction.response.send_message(
                "Owner-only control.", ephemeral=True
            )
            return False

    class PageView(OwnerView):
        def __init__(self, pages: list[str]) -> None:
            super().__init__(timeout=900)
            self.pages = pages or ["No rows available."]
            self.index = 0

        async def _show(self, interaction: Any) -> None:
            await interaction.response.edit_message(
                content=(
                    f"{self.pages[self.index]}\n\n"
                    f"Page {self.index + 1}/{len(self.pages)}"
                ),
                view=self,
            )

        @discord.ui.button(label="Previous", style=discord.ButtonStyle.secondary)
        async def previous(self, interaction: Any, _button: Any) -> None:
            self.index = (self.index - 1) % len(self.pages)
            await self._show(interaction)

        @discord.ui.button(label="Next", style=discord.ButtonStyle.secondary)
        async def next(self, interaction: Any, _button: Any) -> None:
            self.index = (self.index + 1) % len(self.pages)
            await self._show(interaction)

    return OwnerView, PageView
