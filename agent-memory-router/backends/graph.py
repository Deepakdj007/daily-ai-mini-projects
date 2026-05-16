"""
backends/graph.py
-----------------
Simplified knowledge graph backend using a plain Python dict.
In a real system, this connects to Neo4j or LightRAG.
Each key is an entity; its value is a list of (relationship, target) tuples.
"""

# A minimal in-memory graph: entity → [(relationship, target_entity)]
GRAPH = {
    "Project Apollo": [
        ("was_delayed_by", "Supply chain issues in Q2"),
        ("impacted", "Q3 APAC Revenue"),
    ],
    "Q3 APAC Revenue": [
        ("was_below_target_by", "18%"),
        ("reviewed_by", "Priya Nair (CFO)"),
    ],
    "Supply chain issues in Q2": [
        ("caused_by", "Vendor dependency on Taiwan fabs"),
    ],
}

def query_graph(user_query: str) -> str:
    """
    Very simplified entity-matching graph traversal.
    Looks for known entity names in the query string and returns
    their direct and 1-hop connections as a readable string.
    """
    results = []

    for entity, relationships in GRAPH.items():
        # Check if the entity name appears in the query
        if entity.lower() in user_query.lower():
            results.append(f"Entity: {entity}")
            for rel, target in relationships:
                results.append(f"  → {rel}: {target}")
                # 1-hop: also retrieve the target's connections
                if target in GRAPH:
                    for sub_rel, sub_target in GRAPH[target]:
                        results.append(f"      → {sub_rel}: {sub_target}")

    if not results:
        return "No matching entities found in the knowledge graph."

    return "\n".join(results)