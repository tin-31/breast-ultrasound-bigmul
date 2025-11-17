import plotly.graph_objects as go

def gauge_pfinal(p_final: float):
    """Gauge (đồng hồ) cho p_final – hiển thị sắc nét trong Dark mode."""
    val = float(np.clip(p_final, 0, 1)) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={'suffix': '%', 'font': {'size': 34, 'color': '#e5e7eb'}},  # chữ sáng, dễ đọc
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#9ca3af'},
                'bar': {'color': '#ef4444'},  # kim đỏ
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, 15],  'color': '#064e3b'},
                    {'range': [15, 30], 'color': '#065f46'},
                    {'range': [30, 60], 'color': '#78350f'},
                    {'range': [60, 85], 'color': '#7f1d1d'},
                    {'range': [85,100], 'color': '#991b1b'},
                ],
                'threshold': {'line': {'color':'#ffffff','width':3}, 'thickness': 0.8, 'value': val}
            }
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb')
    )
    return fig
