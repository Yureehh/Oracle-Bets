"""Shared owner-only Discord view factories."""

from __future__ import annotations

from typing import Any


async def require_owner(interaction: Any, owner_id: int) -> bool:
    """Protect modal callbacks that do not inherit a View interaction check."""
    if int(interaction.user.id) == owner_id:
        return True
    if not interaction.response.is_done():
        await interaction.response.send_message("Owner-only control.", ephemeral=True)
    return False


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


def paginate_select(
    discord: Any, view: Any, selection: Any, options: list[Any], *, label: str, row: int
) -> None:
    """Keep every option reachable within Discord's 25-option limit."""
    index = 0
    pages = [options[start : start + 25] for start in range(0, len(options), 25)]
    selection.options = pages[0]
    if len(pages) == 1:
        return
    for direction, offset in (("Previous", -1), ("Next", 1)):
        button = discord.ui.Button(
            label=f"{direction} {label}", style=discord.ButtonStyle.secondary, row=row
        )

        def callback(step: int) -> Any:
            async def change(interaction: Any) -> None:
                nonlocal index
                index = (index + step) % len(pages)
                selection.options = pages[index]
                await interaction.response.edit_message(view=view)

            return change

        button.callback = callback(offset)
        view.add_item(button)
