"""Kompatibilitas entry point lama dengan paket llmname."""

from llmname.eda.reporting import build_report as build_report
from llmname.eda.reporting import build_summary as build_summary
from llmname.eda.reporting import write_reports as write_reports
from llmname.eda.reporting.tables import (
    LOGGER as LOGGER,
)
from llmname.eda.reporting.tables import (
    _compact_openings as _compact_openings,
)
from llmname.eda.reporting.tables import (
    _compact_terms as _compact_terms,
)
from llmname.eda.reporting.tables import (
    _figure_markdown as _figure_markdown,
)
from llmname.eda.reporting.tables import (
    _format_decimal as _format_decimal,
)
from llmname.eda.reporting.tables import (
    _format_int as _format_int,
)
from llmname.eda.reporting.tables import (
    _integrity_table as _integrity_table,
)
from llmname.eda.reporting.tables import (
    _top_feature_bullets as _top_feature_bullets,
)
from llmname.eda.reporting.tables import (
    collect_versions as collect_versions,
)
from llmname.eda.reporting.tables import (
    markdown_table as markdown_table,
)
