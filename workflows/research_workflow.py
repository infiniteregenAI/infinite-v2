import json
from typing import Optional, Iterator, List

from pydantic import BaseModel, Field

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.utils.log import logger

# Define the reserved agents to be used exclusively for research work
from utils.reserved_agents import climate_ai, green_pill_ai, owocki_ai, gitcoin_ai

class ResearchSource(BaseModel):
    title: str = Field(..., description="Title of the research source")
    url: str = Field(..., description="URL of the source")
    authors: Optional[List[str]] = Field(default=None, description="Authors of the source")
    publication_date: Optional[str] = Field(default=None, description="Publication date")
    summary: str = Field(..., description="Brief summary of the source")
    key_findings: List[str] = Field(..., description="Key findings or points from the source")
    methodology: Optional[str] = Field(default=None, description="Research methodology used, if applicable")

class ResearchResults(BaseModel):
    sources: list[ResearchSource]
    topic_overview: str = Field(..., description="Brief overview of the research topic")

class ResearchWorkflow(Workflow):
    researcher: List[Agent] = [climate_ai, green_pill_ai, owocki_ai, gitcoin_ai]  # Limit to four reserved agents

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        """Execute the research workflow."""
        logger.info(f"Starting research on: {topic}")

        # Check cache first
        if use_cache:
            cached_report = self.get_cached_report(topic)
            if cached_report:
                yield RunResponse(content=cached_report, event=RunEvent.workflow_completed)
                return

        try:
            research_results = self.gather_research(topic)
            if not research_results:
                yield RunResponse(
                    event=RunEvent.workflow_completed,
                    content=f"Unable to find sufficient research sources for: {topic}"
                )
                return
            
            # Format results and yield
            formatted_results = self.format_results_as_markdown(research_results)
            self.add_report_to_cache(topic, formatted_results)
            yield RunResponse(content=formatted_results, event=RunEvent.workflow_completed)
        except Exception as e:
            logger.error(f"Error in research workflow: {str(e)}")
            yield RunResponse(content=f"An error occurred: {str(e)}", event=RunEvent.workflow_completed)


    def gather_research(self, topic: str) -> Optional[ResearchResults]:
        """Gather research collectively from all agents and synthesize into a single output."""
        combined_insights = []
        
        try:
            # Send the research topic to all agents simultaneously
            responses = [
                agent.run(f"Collaboratively analyze and provide a comprehensive response on: {topic}")
                for agent in self.researcher
            ]
            
            # Collect and combine their insights into a single list
            for response in responses:
                logger.info(f"Agent response: {response.content}")
                combined_insights.append(response.content.strip())
            
            # Merge all responses into one cohesive result
            unified_response = "\n".join(combined_insights)
            
            # Create a single ResearchResults object with unified data
            return ResearchResults(
                sources=[],  # No need for individual source tracking
                topic_overview=unified_response
            )
        except Exception as e:
            logger.error(f"Error during collaborative research: {str(e)}")
            return None


    def format_results_as_markdown(self, results: ResearchResults) -> str:
        """Format the unified research results into a single markdown report."""
        markdown = f"# Research on {results.topic_overview}\n\n"
        markdown += results.topic_overview  # Include the unified content directly
        return markdown


    def get_cached_report(self, topic: str) -> Optional[str]:
        """Retrieve cached research report."""
        return self.session_state.get("research_reports", {}).get(topic)

    def add_report_to_cache(self, topic: str, report: str):
        """Cache the research report."""
        self.session_state.setdefault("research_reports", {})
        self.session_state["research_reports"][topic] = report

if __name__ == "__main__":
    from rich.prompt import Prompt

    # Get research topic from user
    topic = Prompt.ask(
        "[bold]Enter a research topic[/bold]\n✨",
        default="Impact of Artificial Intelligence on Climate Change Mitigation",
    )

    # Create URL-safe topic string
    url_safe_topic = topic.lower().replace(" ", "-")

    # Initialize the research workflow
    research_workflow = ResearchWorkflow(
        session_id=f"research-on-{url_safe_topic}",
        storage=SqlWorkflowStorage(
            table_name="research_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    # Execute the workflow
    for response in research_workflow.run(topic=topic, use_cache=True):
        print(response.content)
