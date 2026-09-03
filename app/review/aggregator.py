"""
Finding aggregator for merging and ranking findings from all review agents.

After all five agents run in parallel, their findings are combined into a
single list.

The aggregator:
1. Groups findings by file, line, and category.
2. Keeps the highest-confidence finding from each group.
3. Boosts confidence when multiple agents report the same issue.
4. Removes duplicate findings with the same file and normalized title.
5. Sorts findings by severity and confidence.
"""

import logging
from collections import defaultdict

from app.review.state import ReviewFindingDict


logger = logging.getLogger(__name__)


SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def aggregate_findings(
    all_findings: list[ReviewFindingDict],
) -> list[ReviewFindingDict]:
    """
    Deduplicate, merge, and rank findings from all review agents.

    Deduplication strategy:
        - Same file_path + line_number + category:
          keep the finding with the highest confidence.
        - If multiple agents report the same finding:
          slightly increase confidence.
        - Same file_path + normalized title:
          keep only the first finding.

    Args:
        all_findings:
            Raw findings returned by all review agents.

    Returns:
        A deduplicated and severity-ranked list of findings.
    """

    if not all_findings:
        return []

    # ---------------------------------------------------------------
    # Step 1: Group findings by file, line, and category.
    # ---------------------------------------------------------------

    finding_groups: dict[
        tuple[str, int | None, str],
        list[ReviewFindingDict],
    ] = defaultdict(list)

    for finding in all_findings:
        group_key = (
            finding.get("file_path", ""),
            finding.get("line_number"),
            finding.get("category", "quality"),
        )

        finding_groups[group_key].append(finding)

    # ---------------------------------------------------------------
    # Step 2: Merge findings within each group.
    # ---------------------------------------------------------------

    merged_findings: list[ReviewFindingDict] = []

    for finding_group in finding_groups.values():
        # Highest-confidence finding becomes the primary finding.
        finding_group.sort(
            key=lambda finding: finding.get("confidence", 0.0),
            reverse=True,
        )

        best_finding = dict(finding_group[0])

        # Collect the names of agents that reported this issue.
        agent_names = list(
            dict.fromkeys(
                finding.get("agent", "unknown")
                for finding in finding_group
            )
        )

        # Multiple agents agreeing increases confidence.
        if len(agent_names) > 1:
            additional_agent_count = len(agent_names) - 1

            best_finding["confidence"] = min(
                1.0,
                best_finding.get("confidence", 0.0)
                + (0.05 * additional_agent_count),
            )

            confirmed_by = ", ".join(agent_names)

            best_finding["description"] = (
                f"{best_finding['description']}\n"
                f"[Confirmed by: {confirmed_by}]"
            )

        merged_findings.append(best_finding)

    # ---------------------------------------------------------------
    # Step 3: Remove duplicate titles within the same file.
    #
    # This is exact normalized matching, not fuzzy title similarity.
    # ---------------------------------------------------------------

    seen_finding_titles: set[str] = set()
    deduplicated_findings: list[ReviewFindingDict] = []

    for finding in merged_findings:
        normalized_title = str(finding.get("title", "")).lower().strip()

        title_key = (
            f"{finding.get('file_path', '')}:"
            f"{normalized_title}"
        )

        if title_key in seen_finding_titles:
            continue

        seen_finding_titles.add(title_key)
        deduplicated_findings.append(finding)

    # ---------------------------------------------------------------
    # Step 4: Sort by severity first, then confidence.
    # ---------------------------------------------------------------

    deduplicated_findings.sort(
        key=lambda finding: (
            SEVERITY_ORDER.get(finding.get("severity", "low"), 4),
            -finding.get("confidence", 0.0),
        )
    )

    logger.info(
        "Aggregated %d raw findings into %d final findings",
        len(all_findings),
        len(deduplicated_findings),
    )

    return deduplicated_findings