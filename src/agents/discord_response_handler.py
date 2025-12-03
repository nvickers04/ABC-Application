#!/usr/bin/env python3
"""
Discord Response Handler - Handles sending agent responses to Discord
Separated from Live Workflow Orchestrator for better modularity and maintainability.
"""

import asyncio
import discord
import os
import logging
from typing import Dict, List, Any, Optional, cast
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DiscordResponseHandler:
    """
    Handles sending agent responses and workflow updates to Discord channels.
    Separated from the main workflow orchestrator for better separation of concerns.
    """

    def __init__(self):
        # Discord client and channels
        self.client = None
        self.channel = None  # General channel for summaries
        self.alerts_channel = None  # Dedicated channel for trade alerts
        self.ranked_trades_channel = None  # Dedicated channel for ranked trade proposals
        self.commands_channel = None  # Dedicated channel for command documentation

        # Discord readiness
        self.discord_ready = asyncio.Event()

    def set_client(self, client):
        """Set the Discord client (shared from orchestrator)."""
        self.client = client

    async def _setup_discord_channels(self):
        """Set up specialized Discord channels."""
        if not self.client:
            return

        guild_id = int(os.getenv('DISCORD_GUILD_ID', '0'))
        if not guild_id:
            logger.warning("DISCORD_GUILD_ID not set")
            return

        guild = self.client.get_guild(guild_id)
        if not guild:
            logger.warning(f"Could not find guild with ID {guild_id}")
            return

        # Set up general channel
        general_channel_id = os.getenv('DISCORD_GENERAL_CHANNEL_ID')
        if general_channel_id:
            try:
                self.channel = guild.get_channel(int(general_channel_id))
                if self.channel:
                    logger.info(f"General channel configured: #{self.channel.name}")
            except ValueError:
                logger.warning(f"Invalid general channel ID: {general_channel_id}")

        # Set up alerts channel
        alerts_channel_id = os.getenv('DISCORD_ALERTS_CHANNEL_ID')
        if alerts_channel_id:
            try:
                self.alerts_channel = guild.get_channel(int(alerts_channel_id))
                if self.alerts_channel:
                    logger.info(f"Alerts channel configured: #{self.alerts_channel.name}")
            except ValueError:
                logger.warning(f"Invalid alerts channel ID: {alerts_channel_id}")

        # Set up ranked trades channel
        ranked_trades_channel_id = os.getenv('DISCORD_RANKED_TRADES_CHANNEL_ID')
        if ranked_trades_channel_id:
            try:
                self.ranked_trades_channel = guild.get_channel(int(ranked_trades_channel_id))
                if self.ranked_trades_channel:
                    logger.info(f"Ranked trades channel configured: #{self.ranked_trades_channel.name}")
            except ValueError:
                logger.warning(f"Invalid ranked trades channel ID: {ranked_trades_channel_id}")

    async def send_agent_responses(self, responses: List[Dict[str, Any]], phase_key: str):
        """
        Send agent responses to Discord in a professional format.

        Args:
            responses: List of agent response dictionaries
            phase_key: The phase identifier for the responses
        """
        if not self.discord_ready.is_set() or not self.channel:
            logger.warning("Discord not ready or channel not configured - skipping response send")
            return

        try:
            await self._present_agent_responses_enhanced(self.channel, responses, phase_key)
        except Exception as e:
            logger.error(f"Failed to send agent responses to Discord: {e}")

    async def send_workflow_status(self, message: str):
        """
        Send workflow status updates to Discord.

        Args:
            message: Status message to send
        """
        if not self.discord_ready.is_set() or not self.channel:
            return

        try:
            await self.channel.send(message)
        except Exception as e:
            logger.error(f"Failed to send workflow status: {e}")

    async def send_trade_alert(self, alert_message: str, alert_type: str = "trade"):
        """
        Send trade alerts to the appropriate Discord channel.

        Args:
            alert_message: The alert message
            alert_type: Type of alert (trade, alert, etc.)
        """
        if not self.discord_ready.is_set():
            return

        target_channel = self.alerts_channel if self.alerts_channel else self.channel
        if not target_channel:
            return

        try:
            formatted_alert = f"🚨 **TRADE ALERT** 🚨\n{alert_message}"
            await target_channel.send(formatted_alert)
        except Exception as e:
            logger.error(f"Failed to send trade alert: {e}")

    async def send_ranked_trade_info(self, trade_message: str, trade_type: str = "proposal"):
        """
        Send ranked trade proposals to the appropriate Discord channel.

        Args:
            trade_message: The trade proposal message
            trade_type: Type of trade info (proposal, execution, etc.)
        """
        if not self.discord_ready.is_set():
            return

        target_channel = self.ranked_trades_channel if self.ranked_trades_channel else self.channel
        if not target_channel:
            return

        try:
            formatted_trade = f"📊 **RANKED TRADE {trade_type.upper()}** 📊\n{trade_message}"
            await target_channel.send(formatted_trade)
        except Exception as e:
            logger.error(f"Failed to send ranked trade info: {e}")

    async def _present_agent_responses_enhanced(self, channel: discord.TextChannel, responses: List[Dict[str, Any]], phase_key: str):
        """Present agent responses in a professional format with logical segmentation"""
        logger.info(f"_present_agent_responses_enhanced called with {len(responses)} responses for phase {phase_key}")
        if not responses:
            return

        # Verify channel permissions
        if not channel.permissions_for(channel.guild.me).send_messages:
            logger.error("Bot does not have send_messages permission in the channel")
            return

        # Group responses by agent
        agent_summaries = {}
        detailed_responses = []

        for response_data in responses:
            agent_name = response_data['agent']
            response = response_data['response']
            command = response_data.get('command', '')

            # Initialize agent summary if not exists
            if agent_name not in agent_summaries:
                agent_summaries[agent_name] = []

            # Sanitize agent output before display
            sanitized_response = self._sanitize_agent_output(response)

            # Create professional agent summary
            if isinstance(sanitized_response, dict):
                response_dict = cast(Dict[str, Any], sanitized_response)
                if 'analysis_type' in response_dict:
                    summary = f"{agent_name.title()} Agent: {response_dict.get('analysis_type', 'analysis')} analysis"
                    if 'confidence_level' in response_dict:
                        summary += f" (Confidence: {response_dict['confidence_level']})"
                elif 'error' in response_dict:
                    summary = f"{agent_name.title()} Agent: Working through {response_dict['error']}"
                else:
                    summary = f"{agent_name.title()} Agent: Analysis complete"
            else:
                summary = f"{agent_name.title()} Agent: {str(sanitized_response)[:100]}..."

            agent_summaries[agent_name].append(summary)

            # Prepare detailed response with professional formatting
            if isinstance(sanitized_response, dict):
                detailed = f"**{agent_name.title()} Agent Analysis**\n"
                detailed += f"Command: {command}\n"
                for key, value in sanitized_response.items():
                    if key not in ['timestamp', 'agent_role']:
                        formatted_value = self._format_response_value(key, value)
                        emoji_map = {
                            'trade_proposals': '📈',
                            'confidence_level': '🎯',
                            'risk_assessment': '⚠️',
                            'analysis': '🔍',
                            'recommendations': '💡'
                        }
                        emoji = emoji_map.get(key, '•')
                        detailed += f"{emoji} **{key.replace('_', ' ').title()}:** {formatted_value}\n"
                detailed_responses.append(detailed.rstrip())
            else:
                formatted_response = self._format_text_response_professionally(str(sanitized_response))
                detailed_responses.append(f"**{agent_name.title()} Agent:** {formatted_response}")

        # Send professional summary first
        summary_text = f"Collaborative analysis complete: {len(responses)} responses from {len(agent_summaries)} agents\n"
        for agent_name, summaries in agent_summaries.items():
            summary_text += f"• **{agent_name.title()}:** {len(summaries)} contributions\n"
        try:
            await channel.send(summary_text)
            logger.info("Successfully sent agent response summary to Discord")
        except Exception as e:
            logger.error(f"Failed to send agent response summary to Discord: {e}")
        await asyncio.sleep(1)

        # Send detailed responses
        for detailed in detailed_responses:
            segments = self._break_into_logical_segments(detailed)
            for segment in segments:
                if segment.strip():
                    try:
                        await channel.send(segment)
                        logger.debug("Successfully sent agent response segment to Discord")
                    except Exception as e:
                        logger.error(f"Failed to send agent response segment to Discord: {e}")
                    await asyncio.sleep(1)

        # Check for trade-related content and send alerts
        for response_data in responses:
            response = response_data['response']
            if isinstance(response, dict):
                response_text = str(response)
            else:
                response_text = str(response)

            if self._is_trade_related_message(response_text):
                alert_content = self._extract_trade_alert_info(response_data)
                if alert_content:
                    await self.send_trade_alert(alert_content, "trade")

        # Check for ranked trade proposals
        for response_data in responses:
            response = response_data['response']
            agent_name = response_data.get('agent', 'Unknown')

            if isinstance(response, dict):
                if 'trade_proposals' in response:
                    proposals = response['trade_proposals']
                    if isinstance(proposals, list) and proposals:
                        ranked_proposals = self._rank_trade_proposals(proposals)
                        ranked_trade_message = f"**{agent_name.title()} Agent** presented {len(ranked_proposals)} ranked trade proposal(s):\n\n"
                        for i, proposal in enumerate(ranked_proposals[:5], 1):
                            if isinstance(proposal, dict):
                                instrument = proposal.get('instrument', 'Unknown')
                                action = proposal.get('action', 'Unknown')
                                confidence = proposal.get('confidence', 'Unknown')
                                reasoning = proposal.get('reasoning', 'No reasoning provided')

                                ranked_trade_message += f"**#{i} {action.upper()} {instrument}**\n"
                                ranked_trade_message += f"• Confidence: {confidence}\n"
                                ranked_trade_message += f"• Reasoning: {reasoning[:200]}...\n\n"

                        await self.send_ranked_trade_info(ranked_trade_message, "proposal")

        await channel.send("Parallel collaboration complete.")
        await asyncio.sleep(1)

    def _sanitize_agent_output(self, response: Any) -> Any:
        """Sanitize agent output for safe display."""
        if isinstance(response, str):
            # Remove potentially harmful content
            return response.replace('`', '').replace('@', '@\u200b')
        return response

    def _format_response_value(self, key: str, value: Any) -> str:
        """Format response values for display."""
        if isinstance(value, (list, dict)):
            return str(value)[:500] + "..." if len(str(value)) > 500 else str(value)
        return str(value)

    def _format_text_response_professionally(self, text: str) -> str:
        """Format text responses with professional segmentation."""
        return text[:1000] + "..." if len(text) > 1000 else text

    def _break_into_logical_segments(self, text: str) -> List[str]:
        """Break text into logical segments for Discord messages."""
        segments = []
        max_length = 1800  # Leave room for formatting

        while text:
            if len(text) <= max_length:
                segments.append(text)
                break

            # Try to break at paragraph or sentence boundaries
            chunk = text[:max_length]
            last_newline = chunk.rfind('\n')
            last_period = chunk.rfind('. ')
            last_space = chunk.rfind(' ')

            if last_newline > max_length * 0.7:
                cut_point = last_newline
            elif last_period > max_length * 0.7:
                cut_point = last_period + 1
            elif last_space > max_length * 0.7:
                cut_point = last_space
            else:
                cut_point = max_length

            segments.append(text[:cut_point].rstrip())
            text = text[cut_point:].lstrip()

        return segments

    def _is_trade_related_message(self, message: str) -> bool:
        """Check if a message contains trade-related content."""
        trade_keywords = ['buy', 'sell', 'trade', 'position', 'execute', 'entry', 'exit']
        return any(keyword in message.lower() for keyword in trade_keywords)

    def _extract_trade_alert_info(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract trade alert information from agent response."""
        response = response_data.get('response', '')
        agent_name = response_data.get('agent', 'Unknown')

        if isinstance(response, dict):
            if 'trade_proposals' in response:
                proposals = response['trade_proposals']
                if proposals:
                    first_proposal = proposals[0] if isinstance(proposals, list) else proposals
                    if isinstance(first_proposal, dict):
                        instrument = first_proposal.get('instrument', 'Unknown')
                        action = first_proposal.get('action', 'Unknown')
                        return f"{agent_name.title()} Agent suggests {action.upper()} {instrument}"

        return None

    def _rank_trade_proposals(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank trade proposals by confidence."""
        def get_confidence(proposal):
            if isinstance(proposal, dict):
                conf = proposal.get('confidence', 0)
                if isinstance(conf, str):
                    # Try to parse confidence strings
                    conf_str = conf.lower()
                    if 'high' in conf_str:
                        return 3
                    elif 'medium' in conf_str or 'moderate' in conf_str:
                        return 2
                    elif 'low' in conf_str:
                        return 1
                try:
                    return float(conf)
                except (ValueError, TypeError):
                    pass
            return 0

        return sorted(proposals, key=get_confidence, reverse=True)

    async def shutdown(self):
        """Shutdown the Discord client."""
        if self.client:
            await self.client.close()