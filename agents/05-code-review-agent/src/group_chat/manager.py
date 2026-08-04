"""Builds the GroupChat and GroupChatManager that run the Critic -> Fixer -> Tester cycle.

Two real details worth being explicit about, verified against this AutoGen version's
actual `GroupChat`/`GroupChatManager` source rather than assumed:

1. `user_proxy` is deliberately NOT included in `groupchat.agents`. AutoGen's
   round-robin speaker selection resolves the *next* speaker as
   `agents[(agents.index(last_speaker) + 1) % len(agents)]`, falling back to index -1
   (so index 0) when `last_speaker` isn't in the list. Since `user_proxy` initiates the
   chat but isn't in `groupchat.agents`, that fallback lands on `agents[0]` == Critic --
   exactly the desired starting speaker. If `user_proxy` WERE included, round-robin
   would cycle back to it after every Tester turn, forcing it to "speak" (with
   `human_input_mode="NEVER"`) before returning to Critic -- wasting a turn and
   inserting a spurious 4th participant into every cycle.

2. `GroupChatManager.run_chat` counts EVERY message as one unit of `max_round`,
   including the initiating message from `user_proxy` -- it is not "N groups of 3".
   To guarantee `max_rounds` full Critic/Fixer/Tester cycles complete before AutoGen's
   own max-round cutoff can fire, `groupchat.max_round` is set to `max_rounds * 3 + 1`
   (the `+1` accounts for that initiating message consuming the first slot).
"""

from __future__ import annotations

import autogen

from src.group_chat.termination import make_termination_check


def create_group_chat(
    critic: autogen.AssistantAgent,
    fixer: autogen.AssistantAgent,
    tester: autogen.AssistantAgent,
    user_proxy: autogen.UserProxyAgent,
    max_rounds: int,
) -> tuple[autogen.GroupChat, autogen.GroupChatManager]:
    """Create and configure the group chat.

    Speaker selection order (round_robin over [Critic, Fixer, Tester]):
        Critic reviews -> Fixer addresses findings -> Tester reviews Fixer's code ->
        back to Critic, unless both Critic and Tester have just approved (see
        termination.py), or `max_rounds` full cycles have completed.
    """
    # user_proxy is accepted here (main.py calls user_proxy.initiate_chat(manager, ...)
    # to actually start the conversation) but deliberately excluded from
    # groupchat.agents -- see module docstring for why.
    groupchat = autogen.GroupChat(
        agents=[critic, fixer, tester],
        messages=[],
        max_round=max_rounds * 3 + 1,
        speaker_selection_method="round_robin",
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=critic.llm_config,
        is_termination_msg=make_termination_check(groupchat),
    )

    return groupchat, manager
