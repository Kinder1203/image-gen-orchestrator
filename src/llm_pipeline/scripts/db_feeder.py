from pathlib import Path

from loguru import logger

from ..core.config import config

COLLECTION_NAME = "ring_gemma_rules"
EMBEDDING_MODEL = "nomic-embed-text"

CURATED_RULES = [
    {
        "id": "material_white_metals",
        "category": "Ring_Material",
        "title": "White Metals and Contrast",
        "tags": ["white-gold", "platinum", "silver", "background"],
        "content": (
            "White gold, platinum, and silver read brightest at the band edge. For clean rembg, "
            "pair them with a solid dark background such as black, charcoal, or deep navy. Never "
            "place white metals on white, pale gray, or reflective chrome-like backgrounds."
        ),
    },
    {
        "id": "material_yellow_gold",
        "category": "Ring_Material",
        "title": "Yellow Gold and Cool Backgrounds",
        "tags": ["yellow-gold", "background", "contrast"],
        "content": (
            "Yellow gold needs a cool contrasting background to preserve the band silhouette. "
            "Prefer solid cobalt blue, cool indigo, cyan, or teal backgrounds. Avoid beige, cream, "
            "warm gold, and peach backgrounds that blend into the metal."
        ),
    },
    {
        "id": "material_rose_gold",
        "category": "Ring_Material",
        "title": "Rose Gold and Blue-Green Contrast",
        "tags": ["rose-gold", "background", "contrast"],
        "content": (
            "Rose gold performs best against cool blue-green backgrounds such as teal, cyan, or "
            "cool slate blue. Avoid pink, blush, salmon, or warm brown backgrounds because they "
            "collapse the outline of the band."
        ),
    },
    {
        "id": "material_dark_metals",
        "category": "Ring_Material",
        "title": "Dark Metal Visibility",
        "tags": ["black-metal", "titanium", "background"],
        "content": (
            "Black titanium, gunmetal, and dark oxidized rings require a pale contrasting "
            "background such as icy gray, off-white, or cool light blue. Do not place dark rings "
            "on black or charcoal backgrounds when the output must survive alpha matting."
        ),
    },
    {
        "id": "material_two_tone_priority",
        "category": "Ring_Material",
        "title": "Two-Tone Priority Rule",
        "tags": ["two-tone", "background", "contrast"],
        "content": (
            "For two-tone rings, choose the background against the brightest dominant metal because "
            "losing the brightest edge harms segmentation most. If the ring is mostly white metal "
            "with warm accents, still choose a dark cool background."
        ),
    },
    {
        "id": "gemstone_secondary_priority",
        "category": "Ring_Design",
        "title": "Gemstone Is Secondary to Band Separation",
        "tags": ["gemstone", "composition", "background"],
        "content": (
            "When a gemstone is vivid, the background still must contrast with the ring body first. "
            "Do not sacrifice band-edge separation just to complement the gem. The band silhouette "
            "and inner hole are the primary constraints for rembg and multi-view extraction."
        ),
    },
    {
        "id": "background_single_color",
        "category": "Validation_and_Rembg",
        "title": "Single-Color Background Only",
        "tags": ["background", "rembg", "validation"],
        "content": (
            "Backgrounds for source and edited ring images must be flat and single-color. Avoid "
            "gradients, scenery, props, hands, shadows crossing the band, and textured studio sets. "
            "A clean solid backdrop gives the most reliable alpha and inner-hole extraction."
        ),
    },
    {
        "id": "background_hole_visibility",
        "category": "Validation_and_Rembg",
        "title": "Inner Hole Must Stay Visible",
        "tags": ["inner-hole", "rembg", "validation"],
        "content": (
            "The empty hole inside the ring must remain visually separate from the background after "
            "generation and editing. If the background bleeds into the band opening or reflections "
            "fill the hole, rembg and TRELLIS preparation should reject the image."
        ),
    },
    {
        "id": "lighting_product_style",
        "category": "Ring_Design",
        "title": "Lighting for Product Isolation",
        "tags": ["lighting", "studio", "product-shot"],
        "content": (
            "Use controlled studio lighting with crisp but not blown-out highlights. The ring should "
            "look like an isolated product photo, not a fashion scene. Preserve edge definition and "
            "avoid dramatic colored lighting that contaminates the metal color."
        ),
    },
    {
        "id": "engraving_inner_band",
        "category": "Ring_Customization",
        "title": "Inner Band Engraving",
        "tags": ["engraving", "inner-band"],
        "content": (
            "When the user requests an inside engraving, describe it explicitly as an inner band "
            "engraving. Keep the text short enough to remain legible and ensure the rest of the ring "
            "geometry is preserved."
        ),
    },
    {
        "id": "engraving_outer_band",
        "category": "Ring_Customization",
        "title": "Outer Band Engraving",
        "tags": ["engraving", "outer-band"],
        "content": (
            "Outer band engravings should follow the visible curvature of the ring and remain "
            "subordinate to the base material. Avoid overcrowding the outer surface with text so the "
            "band thickness and finish stay readable."
        ),
    },
    {
        "id": "engraving_legibility",
        "category": "Ring_Customization",
        "title": "Engraving Legibility Rule",
        "tags": ["engraving", "validation"],
        "content": (
            "Engraving requests should be validated for text legibility, placement correctness, and "
            "clean carving edges. If letters are blurry, fused together, or placed on the wrong side "
            "of the ring, the edit should fail validation."
        ),
    },
    {
        "id": "gemstone_addition_precision",
        "category": "Ring_Customization",
        "title": "Gemstone Addition Precision",
        "tags": ["gemstone", "edit", "inpainting"],
        "content": (
            "When adding a stone to an existing ring, the prompt must localize the change and keep "
            "the original band geometry intact. Request the stone type, cut, size impression, and "
            "placement precisely so the model does not redesign the full ring."
        ),
    },
    {
        "id": "edit_preserve_structure",
        "category": "Ring_Customization",
        "title": "Preserve Existing Structure on Edit",
        "tags": ["edit", "structure", "inpainting"],
        "content": (
            "Edits should preserve material, band thickness, silhouette, and overall composition "
            "unless the user explicitly asks to change them. Customization prompts should focus on "
            "the requested delta, not regenerate a completely different ring."
        ),
    },
    {
        "id": "prompt_keyword_style",
        "category": "Gemma_Prompting",
        "title": "Gemma Prompt Formatting",
        "tags": ["prompt", "keywords", "gemma4"],
        "content": (
            "Gemma should output concise comma-separated product-shot keywords rather than chatty "
            "sentences. Mention the ring type, material, finish, engraving or gemstone details, "
            "camera/product style, and the exact solid background color."
        ),
    },
    {
        "id": "prompt_negative_clutter",
        "category": "Gemma_Prompting",
        "title": "Avoid Props and Human Context",
        "tags": ["prompt", "negative-space", "composition"],
        "content": (
            "Source ring images for downstream processing should avoid fingers, hands, jewelry boxes, "
            "fabric, skin, tables, and lifestyle context. Keep the ring isolated and centered so the "
            "workflow remains stable."
        ),
    },
    {
        "id": "multiview_front_angle",
        "category": "Gemma_Multi_Angle_Prompting",
        "title": "Front View Definition",
        "tags": ["multi-view", "front-view"],
        "content": (
            "The front view should showcase the main face of the ring, such as the center stone or "
            "primary top detail. Keep the band centered and readable rather than tilting into a "
            "dramatic perspective shot."
        ),
    },
    {
        "id": "multiview_side_angle",
        "category": "Gemma_Multi_Angle_Prompting",
        "title": "Side View Definition",
        "tags": ["multi-view", "side-view"],
        "content": (
            "The side view should reveal band thickness, gallery structure, prongs, and profile "
            "details. Maintain a product-shot composition instead of cinematic perspective."
        ),
    },
    {
        "id": "multiview_top_angle",
        "category": "Gemma_Multi_Angle_Prompting",
        "title": "Top View Definition",
        "tags": ["multi-view", "top-view"],
        "content": (
            "The top-down view should clearly reveal the band opening and overall symmetry. This view "
            "is especially useful for checking whether the inner hole stays clean after background "
            "removal."
        ),
    },
    {
        "id": "guardrail_input_repair",
        "category": "Validation_and_Rembg",
        "title": "Input Image Repair Strategy",
        "tags": ["guardrail", "input-image", "repair"],
        "content": (
            "If an uploaded ring image lacks sufficient background contrast, repair the background "
            "first through a targeted edit, then continue the requested scenario. This repair loop is "
            "an internal guardrail and should not introduce an extra user approval stop."
        ),
    },
    {
        "id": "validation_background_rule",
        "category": "Validation_and_Rembg",
        "title": "Background Contrast Validation",
        "tags": ["validation", "background", "complementary"],
        "content": (
            "Validation must explicitly reject rings whose background is too close to the metal color. "
            "White ring on white background is always invalid. Warm gold on beige or blush, and dark "
            "metal on black, are also invalid for rembg-oriented workflows."
        ),
    },
    {
        "id": "repair_directive_style",
        "category": "Validation_and_Rembg",
        "title": "Repair Directives Must Be Short",
        "tags": ["validation", "repair", "edit"],
        "content": (
            "When validation fails because of background contrast, produce a short corrective "
            "directive such as 'Change the background to solid pitch black' or 'Replace the backdrop "
            "with solid cool cyan'. Keep the directive direct enough for an inpainting workflow."
        ),
    },
]


def _build_vector_store(persist_directory: str):
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=persist_directory,
    )


def _build_documents() -> tuple[list[str], list[str], list[dict]]:
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc in CURATED_RULES:
        ids.append(doc["id"])
        texts.append(f"[{doc['category']}] {doc['title']}: {doc['content']}")
        metadatas.append(
            {
                "category": doc["category"],
                "title": doc["title"],
                "doc_id": doc["id"],
                "tags": ",".join(doc.get("tags", [])),
            }
        )

    return ids, texts, metadatas


def init_vector_db(reset_collection: bool = True):
    """
    Build the curated Chroma collection used by the ring prompt/validation pipeline.

    By default the feeder refreshes the dedicated collection so repeated runs do not
    stack duplicate documents.
    """

    db_path = Path(config.VECTOR_DB_PATH)
    db_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing Chroma DB at {db_path}...")

    vector_store = _build_vector_store(str(db_path))

    if reset_collection:
        try:
            vector_store.delete_collection()
            logger.info(f"Reset existing '{COLLECTION_NAME}' collection before re-ingestion.")
        except Exception as exc:
            logger.debug(f"Collection reset skipped: {exc}")
        vector_store = _build_vector_store(str(db_path))

    ids, texts, metadatas = _build_documents()
    logger.debug(f"Ingesting {len(texts)} curated ring rules into Vector DB.")
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    logger.success(
        f"Successfully ingested {len(texts)} curated rules into '{COLLECTION_NAME}'."
    )
    logger.info("Ring prompt generation and validation can now retrieve richer domain guidance.")


if __name__ == "__main__":
    init_vector_db()
