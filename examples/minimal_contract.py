"""Credential-free demonstration of MegaMem's dual-view node contract."""

from __future__ import annotations

from megamem.methods import DualNode, validate_batch


def main() -> None:
    node = DualNode(
        node_id="policy:001",
        level="L0",
        tenant_id="demo",
        distilled_text="Travel reimbursement policy and approval rules.",
        detailed_text=(
            "Employees must submit itemized travel receipts within 30 days. "
            "International travel also requires manager approval."
        ),
        distilled_tokens=7,
        detailed_tokens=18,
        source_evidence_ids=["policy:001#section-4"],
        distilled_text_model_alias="example",
        distilled_text_model_status="DEMO",
    )

    report = validate_batch([node])
    if not report["overall_pass"]:
        raise SystemExit(f"Contract validation failed: {report}")

    print("MegaMem contract validation: PASS")
    print(f"Validated nodes: {report['total_nodes']}")
    print(f"Source IDs: {', '.join(node.source_evidence_ids)}")


if __name__ == "__main__":
    main()
