import logging
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from .azure_credential_utils import get_azure_credential
import html
import traceback
from .env_helper import EnvHelper

logger = logging.getLogger(__name__)


class AzureDocumentIntelligenceClient:
    def __init__(self) -> None:
        env_helper: EnvHelper = EnvHelper()

        # Keep the same environment variable names for backward compatibility
        self.AZURE_FORM_RECOGNIZER_ENDPOINT: str = (
            env_helper.AZURE_FORM_RECOGNIZER_ENDPOINT
        )
        if env_helper.AZURE_AUTH_TYPE == "rbac":
            self.document_intelligence_client = DocumentIntelligenceClient(
                endpoint=self.AZURE_FORM_RECOGNIZER_ENDPOINT,
                credential=get_azure_credential(),
                headers={
                    "x-ms-useragent": "chat-with-your-data-solution-accelerator/1.0.0"
                },
            )
        else:
            self.AZURE_FORM_RECOGNIZER_KEY: str = env_helper.AZURE_FORM_RECOGNIZER_KEY

            self.document_intelligence_client = DocumentIntelligenceClient(
                endpoint=self.AZURE_FORM_RECOGNIZER_ENDPOINT,
                credential=AzureKeyCredential(self.AZURE_FORM_RECOGNIZER_KEY),
                headers={
                    "x-ms-useragent": "chat-with-your-data-solution-accelerator/1.0.0"
                },
            )

    form_recognizer_role_to_html = {
        "title": "h1",
        "sectionHeading": "h2",
        "pageHeader": None,
        "pageFooter": None,
        "paragraph": "p",
    }

    def _table_to_html(self, table):
        table_html = "<table>"
        rows = [
            sorted(
                [cell for cell in table.cells if cell.row_index == i],
                key=lambda cell: cell.column_index,
            )
            for i in range(table.row_count)
        ]
        for row_cells in rows:
            table_html += "<tr>"
            for cell in row_cells:
                tag = "th" if cell.kind == "columnHeader" else "td"
                cell_spans = ""
                if cell.column_span is not None and cell.column_span > 1:
                    cell_spans += f' colspan="{cell.column_span}"'
                if cell.row_span is not None and cell.row_span > 1:
                    cell_spans += f' rowspan="{cell.row_span}"'
                table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
            table_html += "</tr>"
        table_html += "</table>"
        return table_html

    def begin_analyze_document_from_url(
        self,
        source_url: str,
        use_layout: bool = False,
    ):
        page_map = []
        offset = 0

        if use_layout:
            model_id = "prebuilt-layout"
        else:
            model_id = "prebuilt-read"

        try:
            logger.info("Method begin_analyze_document_from_url started")
            logger.info("Model ID selected: %s", model_id)

            # Create the analyze request
            analyze_request = AnalyzeDocumentRequest(url_source=source_url)

            poller = self.document_intelligence_client.begin_analyze_document(
                model_id=model_id, analyze_request=analyze_request
            )
            document_intelligence_results = poller.result()

            # (if using layout) mark all the positions of headers
            roles_start = {}
            roles_end = {}
            if hasattr(document_intelligence_results, 'paragraphs') and document_intelligence_results.paragraphs:
                for paragraph in document_intelligence_results.paragraphs:
                    # if paragraph.role!=None:
                    para_start = paragraph.spans[0].offset
                    para_end = paragraph.spans[0].offset + paragraph.spans[0].length
                    roles_start[para_start] = (
                        paragraph.role if paragraph.role is not None else "paragraph"
                    )
                    roles_end[para_end] = (
                        paragraph.role if paragraph.role is not None else "paragraph"
                    )

            for page_num, page in enumerate(document_intelligence_results.pages):
                tables_on_page = []
                if hasattr(document_intelligence_results, 'tables') and document_intelligence_results.tables:
                    tables_on_page = [
                        table
                        for table in document_intelligence_results.tables
                        if table.bounding_regions[0].page_number == page_num + 1
                    ]

                # (if using layout) mark all positions of the table spans in the page
                page_offset = page.spans[0].offset
                page_length = page.spans[0].length
                table_chars = [-1] * page_length
                for table_id, table in enumerate(tables_on_page):
                    for span in table.spans:
                        # replace all table spans with "table_id" in table_chars array
                        for i in range(span.length):
                            idx = span.offset - page_offset + i
                            if idx >= 0 and idx < page_length:
                                table_chars[idx] = table_id

                # build page text by replacing characters in table spans with table html and replace the characters corresponding to headers with html headers, if using layout
                page_text = ""
                added_tables = set()
                for idx, table_id in enumerate(table_chars):
                    if table_id == -1:
                        position = page_offset + idx
                        if position in roles_start.keys():
                            role = roles_start[position]
                            html_role = self.form_recognizer_role_to_html.get(role)
                            if html_role is not None:
                                page_text += f"<{html_role}>"
                        if position in roles_end.keys():
                            role = roles_end[position]
                            html_role = self.form_recognizer_role_to_html.get(role)
                            if html_role is not None:
                                page_text += f"</{html_role}>"

                        page_text += document_intelligence_results.content[page_offset + idx]

                    elif table_id not in added_tables:
                        page_text += self._table_to_html(tables_on_page[table_id])
                        added_tables.add(table_id)

                page_text += " "
                page_map.append(
                    {"page_number": page_num, "offset": offset, "page_text": page_text}
                )
                offset += len(page_text)

            return page_map
        except Exception as e:
            logger.exception("Exception in begin_analyze_document_from_url: %s", e)
            raise ValueError(f"Error: {traceback.format_exc()}. Error: {e}") from e
        finally:
            logger.info("Method begin_analyze_document_from_url ended")
