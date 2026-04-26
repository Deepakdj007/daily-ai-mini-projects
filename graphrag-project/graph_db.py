from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Read connection details from environment
URI = os.environ["NEO4J_URI"]
AUTH = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])

def get_driver():
    """
    Create and return a Neo4j driver instance.
    The driver manages a connection pool — it is safe to create once and reuse.
    """
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()  # Throws if connection fails — catch this early
    return driver

def clear_graph(driver):
    """
    Delete all nodes and relationships from the database.
    Use this before re-ingesting documents to start fresh.
    DETACH DELETE removes nodes AND their connected relationships in one step.
    """
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Graph cleared")

def create_entity_node(driver, entity: dict):
    """
    Create a node for one entity using MERGE (not CREATE).
    MERGE = create if not exists, match if exists. This is idempotent —
    running it twice does not create duplicates.
    """
    with driver.session() as session:
        session.run(
            """
            MERGE (n:Entity {name: $name})
            SET n.type = $type, n.entity_id = $entity_id
            """,
            name=entity["name"],
            type=entity.get("type", "Unknown"),
            entity_id=entity.get("id", "")
        )
        
def create_relationship(driver, relationship: dict, entity_map: dict):
    """
    Create a directed relationship between two entity nodes.
    
    entity_map: a dict mapping the short entity IDs (like "e1") from the 
    extraction step back to the full entity names, so we can look up the 
    correct Neo4j nodes.
    """
    source_id = relationship.get("source")
    target_id = relationship.get("target")
    relation_type = relationship.get("relation", "RELATED_TO")
    
    # Look up the actual entity names from the short IDs
    source_name = entity_map.get(source_id)
    target_name = entity_map.get(target_id)
    
    if not source_name or not target_name:
        # The LLM sometimes references entity IDs that don't exist in its own output.
        # Skip those silently rather than crashing.
        return
    
    # Cypher does not allow dynamic relationship types with parameters.
    # We must use string interpolation for the type name.
    # We sanitize it first: keep only alphanumeric characters and underscores.
    safe_relation = "".join(c for c in relation_type if c.isalnum() or c == "_")
    if not safe_relation:
        safe_relation = "RELATED_TO"
    
    with driver.session() as session:
        session.run(
            f"""
            MATCH (a:Entity {{name: $source_name}})
            MATCH (b:Entity {{name: $target_name}})
            MERGE (a)-[:{safe_relation}]->(b)
            """,
            source_name=source_name,
            target_name=target_name
        )
def load_graph(extracted_data: dict):
    """
    Main ingestion function. Takes the output of extract_from_folder() and
    writes the full graph to Neo4j.
    """
    driver = get_driver()
    
    entities = extracted_data["entities"]
    relationships = extracted_data["relationships"]
    
    # Build a map from short IDs ("e1") to entity names ("OpenAI")
    # This is needed because relationships reference IDs, not names
    entity_map = {e["id"]: e["name"] for e in entities if "id" in e}
    
    print(f"\n⬆Writing {len(entities)} nodes to Neo4j...")
    for entity in entities:
        create_entity_node(driver, entity)
    
    print(f"⬆Writing {len(relationships)} relationships to Neo4j...")
    for rel in relationships:
        create_relationship(driver, rel, entity_map)
    
    print("Graph loaded successfully")
    driver.close()

def get_full_graph(driver) -> dict:
    """
    Fetch all nodes and relationships from Neo4j.
    Returns a dict formatted for the React force-graph component:
    {"nodes": [...], "links": [...]}
    """
    with driver.session() as session:
        # Get all nodes
        node_result = session.run("MATCH (n:Entity) RETURN n.name AS name, n.type AS type")
        nodes = [
            {"id": record["name"], "name": record["name"], "type": record["type"] or "Unknown"}
            for record in node_result
        ]
        
        # Get all relationships
        link_result = session.run(
            "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS source, b.name AS target, type(r) AS relation"
        )
        links = [
            {"source": record["source"], "target": record["target"], "relation": record["relation"]}
            for record in link_result
        ]
    
    return {"nodes": nodes, "links": links}

def get_subgraph(driver, entity_names: list[str], hops: int = 2) -> dict:
    """
    Fetch a subgraph centered on the given entity names.
    hops=2 means: entity → its neighbors → their neighbors.
    This gives enough context for multi-hop reasoning without returning the whole graph.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (start:Entity)
            WHERE start.name IN $names
            MATCH path = (start)-[*1..2]-(connected)
            RETURN DISTINCT start.name AS start_name,
                   connected.name AS conn_name,
                   connected.type AS conn_type
            """,
            names=entity_names
        )
        
        nodes_seen = set()
        nodes = []
        
        for record in result:
            for name, ntype in [(record["start_name"], "Entity"), (record["conn_name"], record["conn_type"])]:
                if name and name not in nodes_seen:
                    nodes_seen.add(name)
                    nodes.append({"name": name, "type": ntype or "Unknown"})
        
        # Get all relationships between the nodes we found
        if nodes_seen:
            rel_result = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.name IN $names AND b.name IN $names
                RETURN a.name AS source, b.name AS target, type(r) AS relation
                """,
                names=list(nodes_seen)
            )
            links = [
                {"source": r["source"], "target": r["target"], "relation": r["relation"]}
                for r in rel_result
            ]
        else:
            links = []
    
    return {"nodes": nodes, "links": links}