import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================================
# SUPPLYGUARD - ML SOFTWARE SUPPLY-CHAIN RISK PLATFORM
# Single-file Streamlit application
# =========================================================

st.set_page_config(
    page_title="SupplyGuard | Software Supply Chain Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- CSS ----------------------------
st.markdown("""
<style>
:root {
    --navy:#06263d;
    --green:#12c985;
    --light:#f5f8fa;
    --text:#102a43;
    --muted:#718096;
}
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}
.stApp {
    background: var(--light);
    color: var(--text);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: var(--navy);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.brand {
    font-size: 28px;
    font-weight: 800;
    padding: 8px 0 20px 0;
}
.brand span {
    color: var(--green);
}
.small-muted {
    color: var(--muted);
    font-size: 13px;
}
.hero {
    background: white;
    border: 1px solid #e3e9ed;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 18px;
}
.hero h1 {
    margin: 0 0 6px 0;
}
.green-text {
    color: #0bb879;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
}
.metric-card {
    background: white;
    border: 1px solid #e3e9ed;
    border-radius: 12px;
    padding: 18px;
    min-height: 120px;
}
.metric-title {
    color: #718096;
    font-size: 13px;
}
.metric-value {
    font-size: 29px;
    font-weight: 700;
    margin: 8px 0;
}
.panel {
    background: white;
    border: 1px solid #e3e9ed;
    border-radius: 12px;
    padding: 20px;
}
.badge-low, .badge-medium, .badge-high, .badge-critical {
    padding: 5px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 12px;
}
.badge-low { background:#e4f8ef; color:#07885a; }
.badge-medium { background:#fff4d6; color:#a66a00; }
.badge-high { background:#ffe5d6; color:#d45c00; }
.badge-critical { background:#ffe1e1; color:#c62828; }
.status-ok { color:#0bb879; font-weight:700; }
.footer {
    background: var(--navy);
    color: white;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 30px;
}
div.stButton > button {
    border-radius: 7px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- ML MODEL --------------------------
@st.cache_resource
def train_risk_model():
    rng = np.random.default_rng(42)
    n = 1800

    dependency_count = rng.integers(1, 250, n)
    known_cves = rng.poisson(1.8, n).clip(0, 12)
    package_age_days = rng.integers(1, 1800, n)
    maintainer_score = rng.uniform(20, 100, n)
    source_trust = rng.uniform(20, 100, n)
    signed_artifact = rng.integers(0, 2, n)
    direct_dependency = rng.integers(0, 2, n)

    risk_score = (
        known_cves * 11
        + dependency_count * 0.10
        + package_age_days * 0.012
        + (100 - maintainer_score) * 0.20
        + (100 - source_trust) * 0.18
        + (1 - signed_artifact) * 17
        + (1 - direct_dependency) * 4
        + rng.normal(0, 7, n)
    )

    labels = np.select(
        [risk_score >= 105, risk_score >= 65],
        ["Critical", "High"],
        default="Low"
    )
    # Add Medium as a realistic middle class
    labels = np.where((risk_score >= 45) & (risk_score < 65), "Medium", labels)

    X = pd.DataFrame({
        "dependency_count": dependency_count,
        "known_cves": known_cves,
        "package_age_days": package_age_days,
        "maintainer_score": maintainer_score,
        "source_trust": source_trust,
        "signed_artifact": signed_artifact,
        "direct_dependency": direct_dependency
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.20, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, model_accuracy = train_risk_model()

# -------------------- SAMPLE DATA -------------------------
@st.cache_data
def get_inventory():
    return pd.DataFrame([
        ["react", "18.2.0", "npm", "npmjs.org", 0, 44, 92, 96, 1, 1],
        ["lodash", "4.17.21", "npm", "npmjs.org", 1, 610, 88, 92, 1, 1],
        ["express", "4.18.2", "npm", "npmjs.org", 0, 510, 90, 94, 1, 1],
        ["log4j-core", "2.17.1", "Maven", "maven-central", 2, 980, 74, 81, 1, 1],
        ["openssl", "3.0.2", "APT", "ubuntu", 1, 300, 70, 79, 1, 1],
        ["unknown-lib", "0.9.1", "npm", "unknown", 4, 1250, 38, 42, 0, 0],
        ["axios", "1.7.2", "npm", "npmjs.org", 0, 120, 91, 95, 1, 1],
        ["spring-core", "6.1.5", "Maven", "maven-central", 1, 190, 94, 96, 1, 1],
    ], columns=[
        "Name", "Version", "Type", "Source", "Known CVEs",
        "Package Age", "Maintainer Score", "Source Trust",
        "Signed", "Direct"
    ])

inventory = get_inventory()

# Calculate ML risk for inventory
def predict_rows(df):
    X = df[[
        "dependency_count", "known_cves", "package_age_days",
        "maintainer_score", "source_trust", "signed_artifact",
        "direct_dependency"
    ]]
    return model.predict(X)

# Sample dependency count derived for demo purposes
inventory["dependency_count"] = [18, 42, 23, 75, 55, 115, 14, 61]
inventory["Risk"] = predict_rows(inventory.rename(columns={
    "Package Age":"package_age_days",
    "Known CVEs":"known_cves",
    "Maintainer Score":"maintainer_score",
    "Source Trust":"source_trust",
    "Signed":"signed_artifact",
    "Direct":"direct_dependency"
}))
inventory["Status"] = np.where(inventory["Risk"].isin(["Critical", "High"]), "Action Required", "Healthy")

# --------------------- SIDEBAR ----------------------------
st.sidebar.markdown('<div class="brand">🛡️ Supply<span>Guard</span></div>', unsafe_allow_html=True)
st.sidebar.caption("SECURE SOFTWARE SUPPLY CHAIN")

menu = st.sidebar.radio(
    "MAIN MENU",
    [
        "📊 Dashboard", "📦 Inventory", "📋 SBOM",
        "⚠️ Vulnerabilities", "🔐 Provenance",
        "📈 ML Risk Analysis", "🛡️ Policy Engine",
        "🚨 Incidents", "📄 Reports", "⚙️ Settings"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("System Healthy")
st.sidebar.caption("ML Risk Engine: Online")
st.sidebar.caption(f"Model validation accuracy: {model_accuracy:.1%}")

# ---------------------- HELPERS ---------------------------
def risk_badge(value):
    cls = {
        "Low": "badge-low",
        "Medium": "badge-medium",
        "High": "badge-high",
        "Critical": "badge-critical"
    }.get(value, "badge-low")
    return f'<span class="{cls}">{value}</span>'

# ---------------------- DASHBOARD -------------------------
if menu == "📊 Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="green-text">SECURE SOFTWARE SUPPLY CHAIN</div>
        <h1>Welcome back, Leelavathi 👋</h1>
        <div class="small-muted">
            Monitor components, vulnerabilities, provenance and ML-based supply-chain risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    total = len(inventory)
    vulnerabilities = int(inventory["Known CVEs"].sum())
    high_risk = int(inventory["Risk"].isin(["High", "Critical"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    for col, title, value, sub in [
        (c1, "Total Components", total, "Live inventory"),
        (c2, "Known Vulnerabilities", vulnerabilities, "Across components"),
        (c3, "High/Critical Risk", high_risk, "ML detected"),
        (c4, "ML Model", f"{model_accuracy:.1%}", "Validation accuracy")
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="small-muted">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="panel"><h3>ML Risk Distribution</h3>', unsafe_allow_html=True)
        risk_counts = inventory["Risk"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low"], fill_value=0
        ).reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.bar(
            risk_counts, x="Risk", y="Count", text="Count",
            title="Current component risk"
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><h3>Recent Security Alerts</h3>', unsafe_allow_html=True)
        alerts = [
            ("🔴", "Critical/High component detected by ML", "Just now"),
            ("🟠", "Untrusted package source identified", "12 min ago"),
            ("🔵", "SBOM validation completed", "31 min ago"),
            ("🟢", "Provenance verification successful", "1 hr ago")
        ]
        for icon, text, time in alerts:
            st.write(f"{icon} **{text}**")
            st.caption(time)
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Quick Actions")
    a, b, c, d = st.columns(4)
    if a.button("🔍 Run ML Scan", use_container_width=True):
        st.success("Scan completed. Risk predictions refreshed.")
    if b.button("📦 View Inventory", use_container_width=True):
        st.info("Open **Inventory** from the sidebar.")
    if c.button("📄 Generate Report", use_container_width=True):
        st.info("Open **Reports** to download the current CSV report.")
    if d.button("🛡️ Check Policies", use_container_width=True):
        st.info("Open **Policy Engine** to review active policies.")

# ---------------------- INVENTORY -------------------------
elif menu == "📦 Inventory":
    st.markdown("## 📦 Software Inventory")
    st.caption("Components are evaluated using the trained ML risk engine.")

    search = st.text_input("🔎 Search component", placeholder="react, log4j, lodash...")
    risk_filter = st.multiselect(
        "Filter by risk",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )

    view = inventory[inventory["Risk"].isin(risk_filter)].copy()
    if search:
        view = view[view["Name"].str.contains(search, case=False, na=False)]

    display = view[[
        "Name", "Version", "Type", "Source",
        "Known CVEs", "Risk", "Status"
    ]].copy()

    st.dataframe(display, use_container_width=True, hide_index=True)

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Inventory CSV",
        csv,
        "supplyguard_inventory.csv",
        "text/csv"
    )

# ------------------------- SBOM ----------------------------
elif menu == "📋 SBOM":
    st.markdown("## 📋 SBOM Management")
    st.caption("Upload a CSV dependency list or generate a demonstration SBOM.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total SBOMs", "124")
    c2.metric("SPDX", "72")
    c3.metric("CycloneDX", "52")

    uploaded = st.file_uploader(
        "Upload dependency CSV",
        type=["csv"],
        help="CSV should contain at least a package/name column."
    )

    if uploaded:
        sbom = pd.read_csv(uploaded)
        st.success("SBOM file loaded successfully.")
        st.dataframe(sbom, use_container_width=True, hide_index=True)
    else:
        st.info("No file uploaded. Showing current inventory as a demo SBOM.")
        st.dataframe(
            inventory[["Name", "Version", "Type", "Source"]],
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        "⬇️ Download Demo SBOM",
        inventory[["Name", "Version", "Type", "Source"]].to_csv(index=False),
        "supplyguard_sbom.csv",
        "text/csv"
    )

# -------------------- VULNERABILITIES ---------------------
elif menu == "⚠️ Vulnerabilities":
    st.markdown("## ⚠️ Vulnerability Center")
    st.caption("Known CVE counts are used as one of the ML risk features.")

    vuln = inventory[inventory["Known CVEs"] > 0].copy()
    vuln["Severity"] = np.select(
        [vuln["Known CVEs"] >= 3, vuln["Known CVEs"] == 2],
        ["Critical", "High"],
        default="Medium"
    )
    vuln["Status"] = np.where(
        vuln["Severity"].isin(["Critical", "High"]), "Open", "Monitoring"
    )

    st.dataframe(
        vuln[["Name", "Version", "Known CVEs", "Severity", "Status"]],
        use_container_width=True,
        hide_index=True
    )

    if st.button("🔍 Run Vulnerability Scan"):
        with st.spinner("Scanning software inventory..."):
            import time
            time.sleep(1)
        st.success(f"Scan completed. {len(vuln)} components require review.")

# ---------------------- PROVENANCE ------------------------
elif menu == "🔐 Provenance":
    st.markdown("## 🔐 Provenance Verification")
    st.caption("Verify source trust, signatures and artifact integrity.")

    source = st.selectbox(
        "Artifact source",
        ["GitHub", "GitLab", "Internal Registry", "Unknown Registry"]
    )
    signed = st.toggle("Artifact digitally signed", value=True)
    build = st.text_input("Build ID", "BUILD-4587")

    if st.button("🔐 Verify Artifact"):
        source_ok = source != "Unknown Registry"
        if source_ok and signed:
            st.success(f"Integrity Valid ✓ — {build}")
        elif source_ok:
            st.warning("Source is trusted, but artifact is unsigned.")
        else:
            st.error("Untrusted source detected. Review provenance before deployment.")

# ---------------------- ML RISK ---------------------------
elif menu == "📈 ML Risk Analysis":
    st.markdown("## 📈 ML-Based Risk Analysis")
    st.caption(
        "This is a machine-learning demonstration. The model is trained inside the app "
        "using generated sample security data; it is not a live CVE database."
    )

    st.info(f"Random Forest model | validation accuracy: {model_accuracy:.1%}")

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            dependency_count = st.slider("Dependency Count", 1, 250, 50)
            known_cves = st.slider("Known CVEs", 0, 12, 1)
            package_age_days = st.slider("Package Age (days)", 1, 1800, 300)
        with c2:
            maintainer_score = st.slider("Maintainer Trust Score", 0, 100, 80)
            source_trust = st.slider("Source Trust Score", 0, 100, 85)
            signed_artifact = st.selectbox("Signed Artifact", ["Yes", "No"])
        with c3:
            direct_dependency = st.selectbox("Direct Dependency", ["Yes", "No"])
            package_name = st.text_input("Package Name", "my-package")
            run_prediction = st.form_submit_button("🚀 Predict Risk", use_container_width=True)

    if run_prediction:
        X_new = pd.DataFrame([{
            "dependency_count": dependency_count,
            "known_cves": known_cves,
            "package_age_days": package_age_days,
            "maintainer_score": maintainer_score,
            "source_trust": source_trust,
            "signed_artifact": 1 if signed_artifact == "Yes" else 0,
            "direct_dependency": 1 if direct_dependency == "Yes" else 0
        }])

        prediction = model.predict(X_new)[0]
        probabilities = model.predict_proba(X_new)[0]
        classes = model.classes_
        confidence = probabilities[list(classes).index(prediction)]

        st.markdown("---")
        st.subheader(f"Prediction for `{package_name}`")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Risk", prediction)
        r2.metric("Confidence", f"{confidence:.1%}")
        r3.metric("Known CVEs", known_cves)

        if prediction == "Critical":
            st.error("🚨 Critical risk: immediate security review recommended.")
        elif prediction == "High":
            st.warning("⚠️ High risk: investigate before production deployment.")
        elif prediction == "Medium":
            st.warning("🟡 Medium risk: monitor and review security controls.")
        else:
            st.success("🟢 Low risk: no major ML risk signal detected.")

        prob_df = pd.DataFrame({
            "Risk Level": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        fig = px.bar(
            prob_df, x="Risk Level", y="Probability",
            text=prob_df["Probability"].map(lambda x: f"{x:.1%}")
        )
        fig.update_layout(yaxis_tickformat=".0%", height=330)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    feature_names = [
        "Dependency Count", "Known CVEs", "Package Age",
        "Maintainer Score", "Source Trust", "Signed Artifact",
        "Direct Dependency"
    ]
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance, x="Importance", y="Feature",
        orientation="h", title="Random Forest feature importance"
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- POLICY ENGINE -----------------------
elif menu == "🛡️ Policy Engine":
    st.markdown("## 🛡️ Policy Engine")
    st.caption("Example policy-as-code controls for a secure software pipeline.")

    policies = pd.DataFrame([
        ["Block High Risk Packages", "Security", "Active", "Block"],
        ["Enforce Signed Commits", "Integrity", "Active", "Block"],
        ["SBOM Required", "Compliance", "Active", "Warn"],
        ["Unknown Registry Check", "Supply Chain", "Active", "Block"],
    ], columns=["Policy Name", "Type", "Status", "Action"])

    st.dataframe(policies, use_container_width=True, hide_index=True)

    st.subheader("Policy Test")
    risk = st.selectbox("Predicted package risk", ["Low", "Medium", "High", "Critical"])
    signed = st.checkbox("Package is signed", value=True)
    source = st.selectbox("Source", ["Trusted Registry", "Unknown Registry"])

    if st.button("Run Policy Check"):
        violations = []
        if risk in ["High", "Critical"]:
            violations.append("High/Critical package risk")
        if not signed:
            violations.append("Unsigned artifact")
        if source == "Unknown Registry":
            violations.append("Unknown package source")

        if violations:
            st.error("Policy violation detected.")
            for item in violations:
                st.write("•", item)
        else:
            st.success("All active policies passed.")

# ----------------------- INCIDENTS ------------------------
elif menu == "🚨 Incidents":
    st.markdown("## 🚨 Security Incidents")

    high = inventory[inventory["Risk"].isin(["High", "Critical"])]

    if high.empty:
        st.success("No active ML-generated high-risk incidents.")
    else:
        for _, row in high.iterrows():
            st.error(
                f"**{row['Risk']} — {row['Name']} {row['Version']}** | "
                f"Known CVEs: {row['Known CVEs']} | Source: {row['Source']}"
            )

# ------------------------ REPORTS -------------------------
elif menu == "📄 Reports":
    st.markdown("## 📄 Security Reports")

    report = inventory[[
        "Name", "Version", "Type", "Source",
        "Known CVEs", "Risk", "Status"
    ]].copy()

    st.dataframe(report, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Security Report",
        report.to_csv(index=False),
        "supplyguard_security_report.csv",
        "text/csv"
    )

    st.markdown("""
    **Report contents**
    - Software component inventory
    - Known vulnerability counts
    - ML-based risk classification
    - Source and provenance information
    - Recommended review status
    """)

# ----------------------- SETTINGS --------------------------
elif menu == "⚙️ Settings":
    st.markdown("## ⚙️ Platform Settings")

    organization = st.text_input("Organization Name", "TechSecure Ltd")
    platform_url = st.text_input("Platform URL", "https://supplyguard.app")
    timezone = st.selectbox("Timezone", ["IST (UTC+5:30)", "UTC"])

    if st.button("💾 Save Changes"):
        st.success(f"Settings saved for {organization}.")

    st.subheader("System Information")
    st.write("**Application:** SupplyGuard")
    st.write("**ML Engine:** Random Forest")
    st.write("**Framework:** Streamlit")
    st.write("**Status:** Online")

# ------------------------- FOOTER -------------------------
st.markdown("""
<div class="footer">
    <b>🛡️ SupplyGuard</b><br>
    Secure Your Software Supply Chain Today.<br>
    <small>ML-powered security analytics • Streamlit demonstration project • © 2026</small>
</div>
""", unsafe_allow_html=True)
