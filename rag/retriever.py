from models.building import BuildingData
from rag.vector_store import VectorStore


class Retriever:

    def __init__(self, vector_store: VectorStore, n_results: int = 4) -> None:
        self._store = vector_store
        self._n_results = n_results

    def get_context(self, building: BuildingData) -> str:
        """
        Build a semantic query from the building description and return
        retrieved norm chunks joined into a single prompt-ready string.

        Returns an empty string if the vector store has no documents yet.
        """
        if self._store.is_empty():
            return ""

        query = self._build_query(building)
        chunks = self._store.query(query, n_results=self._n_results)

        if not chunks:
            return ""

        useful = [c for c in chunks if self._contains_numeric_data(c)]

        if not useful:
            return ""

        return self._format_context(useful)


    @staticmethod
    def _build_query(building: BuildingData) -> str:
        building_type_ru = {
            "residential": "жилое",
            "commercial": "коммерческое",
            "industrial": "промышленное",
            "infrastructure": "инфраструктурное",
        }.get(building.building_type, building.building_type)

        material_hints = ""
        if building.elements:
            materials = {el.material.lower() for el in building.elements}
            material_hints = ", ".join(materials)

        parts = [
            f"нормативная продолжительность строительства в месяцах",
            f"{building_type_ru} здание {building.floors} этажей",
            f"СП 48 МДС нормы сроки",
        ]

        if material_hints:
            parts.append(material_hints)

        return ", ".join(parts)

    @staticmethod
    def _contains_numeric_data(chunk: str) -> bool:
        return any(char.isdigit() for char in chunk)

    @staticmethod
    def _format_context(chunks: list[str]) -> str:
        """Wrap retrieved chunks in a numbered block for the LLM prompt."""
        numbered = "\n\n".join(
            f"[Норма {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
        )
        return "Релевантные строительные нормы и правила:\n\n" + numbered