import json
import os
import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from utils.my_log import logger


class DocumentSearchResponse(BaseModel):
    thinking: str = Field(description="Your thinking process on which nodes are relevant to the question")
    node_list: list[str] = Field(description="List of relevant node IDs found in the tree")


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are given a question and a tree structure of a document. "
        "Each node contains a node id, node title, and a corresponding summary. "
        "Your task is to find all nodes that are likely to contain the answer to the question. "
        "Return only a JSON object with exactly these fields: {{\"thinking\": \"brief explanation\", \"node_list\": [\"node_id_1\", \"node_id_2\"]}}. "
        "Use only the node IDs that actually exist in the tree. Do not include markdown, prose, or extra keys."
    ),
    (
        "human",
        "Question: {query}\n\n"
        "Document tree structure:\n{tree_json}"
    )
])


def _normalize_node_values(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        raw_value = [raw_value]
    if not isinstance(raw_value, list):
        return []

    node_ids = []
    for item in raw_value:
        if isinstance(item, dict):
            for key in ("node_id", "id", "nodeId"):
                if key in item:
                    node_ids.append(str(item[key]))
                    break
        elif item is not None:
            node_ids.append(str(item))
    return node_ids


def _extract_json_payload(raw_content):
    if raw_content is None:
        return None

    if isinstance(raw_content, dict):
        return raw_content

    text = str(raw_content).strip()
    if not text:
        return None

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for key in ("node_list", "node_ids", "relevant_node_ids"):
        pattern = rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            value_text = match.group(1)
            fragments = re.findall(r'"([^"]+)"|\b([A-Za-z0-9_.:-]+)\b', value_text)
            values = []
            for left, right in fragments:
                values.append(left or right)
            if values:
                return {"thinking": "Selected relevant nodes from the document tree.", key: values}

    return None


def _coerce_document_response(raw_content):
    payload = _extract_json_payload(raw_content)
    if payload is None:
        raise ValueError("No JSON object found in PageIndex response")

    if isinstance(payload, list):
        payload = {"thinking": "Selected relevant nodes from the document tree.", "node_list": payload}

    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected PageIndex payload type: {type(payload).__name__}")

    node_key = None
    for candidate in ("node_list", "node_ids", "relevant_node_ids"):
        if candidate in payload:
            node_key = candidate
            break

    if node_key is None:
        raise ValueError(f"PageIndex payload missing a node list field: {payload}")

    normalized = {
        "thinking": str(payload.get("thinking") or payload.get("explanation") or "Selected relevant nodes from the document tree."),
        "node_list": _normalize_node_values(payload[node_key]),
    }
    return DocumentSearchResponse.model_validate(normalized)


def retrieve_dataset(doc_ids, dataset):
    from pageindex import PageIndexClient
    from utils.model_factories import create_default_model
    from tqdm import tqdm

    if "PAGE_INDEX_API_KEY" not in os.environ:
        raise RuntimeError("missing PAGE_INDEX_API_KEY")

    pi_client = PageIndexClient(api_key=os.environ.get("PAGE_INDEX_API_KEY"))
    llm = create_default_model(max_tokens=2048)

    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]

    logger.info(f"PageIndex: loading {len(doc_ids)} document tree(s)")
    trees = []
    ocr_page_cache = {}

    def get_ocr_pages(doc_id):
        if doc_id not in ocr_page_cache:
            response = pi_client.get_ocr(doc_id, format="page")
            pages = response.get("result")
            if not isinstance(pages, list):
                raise ValueError(f"PageIndex returned invalid OCR pages for {doc_id}")

            page_map = {}
            for page in pages:
                if not isinstance(page, dict) or page.get("page_index") is None:
                    raise ValueError(f"PageIndex returned a page without page_index for {doc_id}")
                markdown = page.get("markdown")
                if isinstance(markdown, str) and markdown.strip():
                    page_map[int(page["page_index"])] = markdown
            ocr_page_cache[doc_id] = page_map
        return ocr_page_cache[doc_id]

    for doc_id in doc_ids:
        if not pi_client.is_retrieval_ready(doc_id):
            raise RuntimeError(f"PageIndex document is not ready: {doc_id}")
        logger.info(f"PageIndex: loading tree {doc_id}")
        summary_tree = pi_client.get_tree(
            doc_id, node_summary=True, include_text=False
        )['result']
        content_tree = pi_client.get_tree(doc_id, include_text=True)['result']
        trees.append((doc_id, summary_tree, content_tree))

    contexts = []
    for query in tqdm(dataset["question"], desc="Retrieving tree"):
        logger.info(f"PageIndex: retrieving contexts for query {query[:80]!r}")
        contexts.append([
            retrieve(
                summary_tree,
                llm,
                query,
                content_tree=content_tree,
                ocr_page_loader=lambda doc_id=doc_id: get_ocr_pages(doc_id),
            )
            for doc_id, summary_tree, content_tree in trees
        ])

    dataset["contexts"] = contexts
    dataset["retrieved_contexts"] = contexts
    return dataset


def _collect_node_content(node_id, node_map, summary_node_map, ocr_page_loader, visited):
    if node_id in visited:
        return []
    visited.add(node_id)

    node = node_map.get(node_id)
    summary_node = summary_node_map.get(node_id)
    if node is None:
        node = summary_node
    if node is None:
        return []

    node_text = node.get("text")
    if isinstance(node_text, str) and node_text.strip():
        return [node_text]

    children = node.get("nodes") or (summary_node or {}).get("nodes", [])
    descendant_texts = []
    for child in children:
        child_id = child.get("node_id")
        if child_id is not None:
            descendant_texts.extend(_collect_node_content(
                str(child_id), node_map, summary_node_map, ocr_page_loader, visited
            ))
    if descendant_texts:
        logger.warning(f"PageIndex node {node_id!r} has no text; using descendant content")
        return descendant_texts

    start_page = node.get("start_index", node.get("page_index"))
    end_page = node.get("end_index")
    if end_page is None:
        end_page = start_page
    if start_page is not None and ocr_page_loader is not None:
        try:
            start_page, end_page = int(start_page), int(end_page)
        except (TypeError, ValueError) as error:
            raise ValueError(f"PageIndex node {node_id!r} has an invalid page range") from error
        pages = ocr_page_loader()
        page_texts = [
            pages[page]
            for page in range(start_page, end_page + 1)
            if page in pages
        ]
        if page_texts:
            logger.warning(f"PageIndex node {node_id!r} has no text; using OCR page content")
            return page_texts

    summary = (
        (summary_node or {}).get("summary")
        or node.get("summary")
        or (summary_node or {}).get("prefix_summary")
        or node.get("prefix_summary")
    )
    if isinstance(summary, str) and summary.strip():
        logger.warning(f"PageIndex node {node_id!r} has no text; using its summary")
        return [summary]

    return []


def retrieve(tree, llm, query, content_tree=None, ocr_page_loader=None):
    import pageindex.utils as utils
    from langchain_core.messages import HumanMessage, AIMessage

    tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])
    tree_json_str = json.dumps(tree_without_text, indent=2)
    node_map = utils.create_node_mapping(content_tree if content_tree is not None else tree)
    summary_node_map = utils.create_node_mapping(tree)

    messages = prompt_template.invoke({
        "query": query,
        "tree_json": tree_json_str
    }).to_messages()

    max_attempts = 3
    for attempt in range(max_attempts):
        is_last_attempt = attempt == max_attempts - 1

        try:
            ai_message = llm.invoke(messages)
            raw_content = getattr(ai_message, "content", str(ai_message))
            response = _coerce_document_response(raw_content)
        except (ValidationError, ValueError, TypeError, AttributeError) as e:
            if is_last_attempt:
                logger.error(f"PageIndex schema validation failed after {max_attempts} attempts: {e}")
                raise e
            logger.warning(f"PageIndex schema validation failed on attempt {attempt + 1}; retrying: {e}")
            raw_content = getattr(e, "llm_output", None) or getattr(e, "observation", None)
            messages.append(AIMessage(content=str(raw_content) if raw_content else "[Invalid JSON / Schema Output]"))
            messages.append(HumanMessage(
                content=f"Your previous response failed JSON/schema validation with this error:\n{e}\n"
                        f"Please correct your response to strictly match the requested JSON schema."
            ))
            continue

        resolved_texts = []
        invalid_ids = []
        missing_content_ids = []

        for node_id in response.node_list:
            if node_id in node_map or node_id in summary_node_map:
                node_contents = _collect_node_content(
                    node_id, node_map, summary_node_map, ocr_page_loader, set()
                )
                if node_contents:
                    resolved_texts.extend(node_contents)
                else:
                    missing_content_ids.append(node_id)
            else:
                invalid_ids.append(node_id)

        if missing_content_ids:
            logger.warning(
                f"PageIndex nodes {missing_content_ids} have no text, summary, "
                "or recoverable descendant/page content"
            )
            if not resolved_texts:
                raise ValueError(
                    f"PageIndex selected nodes {missing_content_ids} without "
                    "text, summary, or recoverable descendant/page content"
                )

        if invalid_ids and not is_last_attempt:
            messages.append(AIMessage(content=response.model_dump_json(indent=2)))
            messages.append(HumanMessage(
                content=f"The node IDs {invalid_ids} do not exist in the document tree. "
                        f"Please review the tree structure and return only valid node IDs."
            ))
            continue

        if invalid_ids:
            logger.warning(f"PageIndex returned invalid node IDs {invalid_ids}; ignoring them")

        logger.info(
            f"PageIndex: selected {len(response.node_list)} node(s) and "
            f"resolved {len(resolved_texts)} content segment(s)"
        )
        return "\n\n".join(resolved_texts)
