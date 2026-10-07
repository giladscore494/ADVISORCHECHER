"""Read-only live smoke test against data.gov.il (at most 6 API calls plus one small download).

Usage:  python scripts/smoke_datagov.py "רכב"
Prints what was discovered, inspected and read. Makes no writes and needs no API key.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datagov  # noqa: E402


def main(query: str) -> int:
    client = datagov.CkanClient(max_calls=6)
    try:
        found = client.package_search(query, rows=3)
        print(f"package_search '{query}': {found['count']} datasets")
        for d in found["results"]:
            print(f"  - {d['title']} [{d['publisher']}] id={d['id']} formats={d['formats']}")
        if not found["results"]:
            return 0
        dataset = client.package_show(found["results"][0]["id"])
        print(f"package_show: {dataset['title']} | updated {dataset['last_updated']} | license {dataset['license']}")
        for r in dataset["resources"][:5]:
            print(f"  - resource {r['id']} {r['format']} datastore_active={r['datastore_active']} {r['name']}")
        if dataset["resources"]:
            out = datagov.read_resource(client, dataset["resources"][0]["id"], query, query, limit=3)
            print(f"read_resource: status={out['status']} via {out.get('retrieval', '-')}")
            print((out.get("records") or "\n".join(out.get("passages", [])) or out.get("reason") or out.get("error", ""))[:800])
    except datagov.CkanError as exc:
        print(f"FAILED ({exc.kind}): {exc}")
        return 1
    finally:
        print(f"API calls: {client.calls}")
        for entry in client.log:
            print(f"  {entry['action']}: ok={entry['ok']} http={entry['http_status']} cached={entry['cached']} {entry['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "רכב"))
