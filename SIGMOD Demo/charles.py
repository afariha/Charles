from LinearTree.LinearTree import LinearTree
import streamlit as st
import pandas as pd
from scipy.stats import f_oneway, pearsonr
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px

def rank_attrs(target_df, categorical_attrs, numerical_attrs, target_attr):    
    condition_attrs = [a for a in categorical_attrs]
    for a in numerical_attrs:
        if target_df[a].nunique()/float(len(target_df[a])) <= 0.1:  # treat as categorical attribute
            condition_attrs.append(a)
    
    transformation_attrs = [a for a in numerical_attrs if a not in condition_attrs]

    transformation_scores = {}
    condition_scores = {}

    # Pearson correlation for transformation attributes
    for attr in transformation_attrs:
        try:
            corr, _ = pearsonr(target_df[attr], target_df[target_attr])
            transformation_scores[attr] = abs(corr)  # Use absolute correlation
        except Exception:
            transformation_scores[attr] = 0

    # ANOVA F-statistic for condition attributes
    for attr in condition_attrs:
        try:
            groups = [group[target_attr].dropna().values for _, group in target_df.groupby(attr)]
            if len(groups) > 1:
                f_stat, _ = f_oneway(*groups)
                condition_scores[attr] = f_stat
            else:
                condition_scores[attr] = 0
        except Exception:
            condition_scores[attr] = 0

    # Sort attributes by score
    ranked_transformation = sorted(transformation_scores, key=transformation_scores.get, reverse=True)
    ranked_condition = sorted(condition_scores, key=condition_scores.get, reverse=True)

    return ranked_condition, ranked_transformation

def compute_change_summary(source_df, target_df, target_attr, condition_attrs, transformation_attrs, max_num_condition_attrs, max_num_transformation_attrs, weight, recompute=False):

    if not recompute:
        return st.session_state.computed_summary
    
    change_summary = pd.DataFrame(columns=[
        "ID",
        "Summary",         
        "Details", 
        "Score", 
        "Accuracy", 
        "Interpretability"
    ])

    idx = 0
    for n in range(1, max_num_condition_attrs+1):
        for c_attrs in combinations(condition_attrs, n):
            for m in range(1, max_num_transformation_attrs+1):
                for t_attrs in combinations(transformation_attrs, m):
                    print('Currently trying:', c_attrs, t_attrs)
                    linear_tree = LinearTree(source_df = source_df,
                                            target_df = target_df,
                                            target_col=target_attr,
                                            max_depth = 3,
                                            potential_split_attributes = c_attrs,
                                            potential_transformation_attributes = t_attrs)
                    
                    summary = []

                    linear_tree.transformation.store(source_df, target_df)
                    interpretability = linear_tree.transformation.interpretability
                    accuracy = linear_tree.transformation.accuracy

                    for ct in linear_tree.transformation.conditional_transformations:
                        summary.append({
                            'condition' : str(ct.partition), 
                            'transformation': str(ct.single_transformation),
                            'accuracy': ct.accuracy,
                            'coverage': ct.partition.cardinality * 100 /len(source_df)
                            })

                    cur_summary = {
                        "ID": idx,
                        "Summary": summary,
                        "Details": "Show",
                        "Score": round((weight * accuracy + (1 - weight) * interpretability)*100,2),
                        "Accuracy": str(round(accuracy*100,2)) + '%',
                        "Interpretability": str(round(interpretability*100,2))  + '%'
                    }
                    if str(cur_summary["Summary"]) not in [str(s["Summary"]) for _, s in change_summary.iterrows()]:
                        idx += 1
                        change_summary.loc[len(change_summary)] = cur_summary

    st.session_state.computed_summary = change_summary 
    return st.session_state.computed_summary

def populate_summary(source_df, target_df, target_attr, condition_attrs, transformation_attrs, max_condition, max_transformation, weight):        
    
    summary_df = compute_change_summary(
        source_df=source_df, 
        target_df=target_df, 
        target_attr=target_attr, 
        condition_attrs=condition_attrs, 
        transformation_attrs=transformation_attrs,
        max_num_condition_attrs=max_condition,
        max_num_transformation_attrs=max_transformation,
        weight=weight,
        recompute=st.session_state.recompute_summary)
    
    st.session_state.recompute_summary = False
    
    col1, col2, col3 = st.columns([1, 1.25, 9.25])
    with col1:
        st.markdown("<div style='padding-top:7px;'>Show the top</div>",unsafe_allow_html=True)

    with col2:
        num_rows_to_show = st.number_input(
            label="num_rows_to_show",
            min_value=1,
            max_value=15,
            value=5,  # default value
            step=1,
            label_visibility="hidden",
            key="num_rows_selector"
        )
    with col3:
        st.markdown("<div style='padding-top:7px;'>summaries</div>",unsafe_allow_html=True)        

    # Header
    header_cols = st.columns([6.2, 1, 1, 1.5, 0.8, 2.5], gap="small") 
    headers = ["Summary", "Score", "Accuracy", "Interpretability", "Details", "Visual"]
    for col, header in zip(header_cols, headers):
        col.markdown(f"<div style='text-align:center; background-color:lightblue;text-color:black; padding:2px;font-weight:bold;border:1px solid black;'>{header}</div>",unsafe_allow_html=True)        
                
    st.markdown("<br/>", unsafe_allow_html=True)

    # Rows
    summary_df = summary_df.sort_values(by=["Score"], ascending=False)[:num_rows_to_show]       
    for idx, row in summary_df.iterrows():                
        # st.markdown("<br>", unsafe_allow_html=True)
        # Summary as nested layout using container
        first = True
        for summary in row["Summary"]:
            st.markdown("<br>", unsafe_allow_html=True)  
            cols = st.columns([2, 0.2, 4, 1, 1, 1.5, 0.8, 2.5], gap="small")                            
            cols = [None] + cols

            with cols[1]:                    
                st.markdown(f"<div style='text-align:left; text-color:black; padding:2px; background:#E6FFE6;margin-bottom:6px;'>{summary['condition']}</div>",unsafe_allow_html=True)
                st.markdown('---')

            with cols[2]:                    
                st.markdown(f"<div style='text-align:center; text-color:black; padding:2px;'>➔</div>",unsafe_allow_html=True)
                

            with cols[3]:
                st.markdown(f"<div style='text-align:left; text-color:black; padding:2px; background:#F5E6FF;margin-bottom:6px;'>{summary['transformation']}</div>",unsafe_allow_html=True)
                st.markdown('---')

            if first:
                with cols[4]:
                    st.markdown(f"<div style='text-align:center; text-color:black; padding:2px;'>{str(row['Score'])}%</div>",unsafe_allow_html=True)
            if first:
                with cols[5]:
                    st.markdown(f"<div style='text-align:center; text-color:black; padding:2px;'>{row['Accuracy']}</div>",unsafe_allow_html=True)

            if first:
                with cols[6]:
                    st.markdown(f"<div style='text-align:center; text-color:black; padding:2px;'>{row['Interpretability']}</div>",unsafe_allow_html=True)

            if first:
                with cols[7]:
                    # Unique key for this button
                    btn_key = f"imgbutton_{row['ID']}"

                    # Draw the button
                    clicked = st.button(" ", key=btn_key)

                    if clicked:
                        if st.session_state.clicked_row == row['ID']:
                            st.session_state.clicked_row = -1
                        else:
                            st.session_state.clicked_row = row['ID']            
                            with cols[8]:                                              
                                data = [] 

                                def wrap_text(text, max_length=30):
                                    if len(text) >= 80:
                                        text = text[:80] + " ... "
                                    words = text.split()
                                    lines = []
                                    current_line = ""

                                    for word in words:
                                        # Check if adding the next word exceeds max_length
                                        if len(current_line) + len(word) + (1 if current_line else 0) <= max_length:
                                            current_line += (" " if current_line else "") + word
                                        else:
                                            lines.append(current_line)
                                            current_line = word
                                    if current_line:
                                        lines.append(current_line)

                                    return "<br>".join(lines)                   
                                
                                for i in range(len(st.session_state.computed_summary.loc[st.session_state.clicked_row]["Summary"])):
                                    cond = "Condition: " + st.session_state.computed_summary.loc[st.session_state.clicked_row]["Summary"][i]["condition"]
                                    acc = st.session_state.computed_summary.loc[st.session_state.clicked_row]["Summary"][i]["accuracy"]
                                    coverage = st.session_state.computed_summary.loc[st.session_state.clicked_row]["Summary"][i]["coverage"]
                                    hover_text = f"{wrap_text(cond)}<br><br>Accuracy: {round(acc * 100, 2)}%"
                                    data.append((hover_text, coverage))

                                df = pd.DataFrame(data, columns=["HoverText", "Coverage"])

                                fig = px.pie(df, values="Coverage", names=[""] * len(df), title="")  # Empty names to suppress default text
                                fig.update_traces(
                                    hoverlabel=dict(
                                        align="left",
                                        bgcolor="white",
                                        bordercolor="black",
                                        font_size=12,
                                        font_family="Arial",
                                        namelength=0,
                                    ),
                                    hovertemplate='%{customdata}<extra></extra>',
                                    customdata=df["HoverText"]
                                )
                                fig.update_layout(
                                    margin=dict(l=0, r=0, t=10, b=0),
                                    showlegend=False,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=150
                                )

                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state.clicked_row}")
                
            first = False
        st.markdown('---')


    
                

def setup_ui():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
        body {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
                
        hr {
            margin-top:10px !important;
        }

        .stFileUploaderFile {
            justify-content: flex-start !important;
            text-align: left !important;
        }
        label[data-testid="stWidgetLabel"] {
        display:block;
        }
        
        div[data-baseweb="slider"] {
            padding-left:8px !important;
        }    
                
        div[class*="imgbutton"] > div > button {
                        background-image: url("https://cdn-icons-png.flaticon.com/512/167/167485.png");
                        background-size: contain;
                        background-color: transparent !important;
                        background-repeat: no-repeat;
                        background-position: center;
                        height: auto;
                        width: 60%;
                        padding: 0;
                        margin-left:20%;
                        border: none;
                        color: transparent;
        }
                
        .stVerticalBlock{
                gap:0px
        }
        
        .stHorizontalBlock{
                gap:0px
        }
        .stMainBlockContainer
        {
            padding: 10px;
            padding-top:0px;
            padding-bottom:0px;
        }
                
        header[data-testid="stHeader"] {
            display: none;
        }
        
        .stApp {
            background-color: #eeeeee;
        }
                
        /* Hide uploader hint text */
        div[data-testid="stFileUploader"] {
            width: 500px !important;
            padding: 0;
            display:contents;
        }       
                

        div[data-testid="stFileUploader"] section {
                background-color: #eeeeee !important;
                padding: 0px !important;
        }
                    
        div[data-testid="stFileUploader"] span {
            display: none;
        }
                
        div[data-testid="stFileUploader"] svg {
            display: none;
        }
        div[data-testid="stFileUploader"] small {
            display: none;
        }
        div[data-testid="stFileUploader"] label[data-testid="stWidgetLabel"] {
            display: none;
        }
                
        div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
            display: none;
        }
                
        div[data-testid="stNumberInput"] label[data-testid="stWidgetLabel"] {
            display: none;
        }
        div[data-testid="stNumberInput"] > div > div{
                width:auto;
        }
                
        div.stButton > button:first-child {
            background-color: #d8b4f8;
            color: black;
            padding: 10px 24px;
            font-size: 16px;
            border: none;
            border-radius: 0px;
            cursor: pointer;
            width: 40%;
            margin-left:30%;
            margin-right:30%;
        }
        div.stButton > button:first-child:active {
            background-color: #c084fc;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            transform: scale(0.98);        
            color: black;
        }
                        
        #detailsbutton div.stButton > button {
            background-color: #FF6347 !important; /* tomato red */
            color: white !important;
            font-weight: bold !important;
            border-radius: 12px !important;
            border: 2px solid #FF4500 !important;
        }

    </style>
    """, unsafe_allow_html=True)
    st.title("ChARLES")
    st.markdown("---")

    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1.2, 1])

    target_attr = None
    filtered_attrs = None
    numerical_attrs = None

    if "show_summary_activated" not in st.session_state:
        st.session_state.show_summary_activated = False
        st.session_state.recompute_summary = True
        st.session_state.clicked_row = -1

    with col1:
        st.markdown("**Upload source dataset**")
        source_file = st.file_uploader("none", type=["csv"], key="source_file_uploader", label_visibility="hidden")
    with col2:
        st.markdown("**Upload target dataset**")
        target_file = st.file_uploader("none", type=["csv"], key="target_file_uploader", label_visibility="hidden")

    if source_file and target_file:
        source_df = pd.read_csv(source_file)
        target_df = pd.read_csv(target_file)
        excluded_attrs = ['Employee_id'] 
        filtered_attrs = sorted(
            [col for col in source_df.columns if col not in excluded_attrs],
            key=lambda x: (
                (0 if x[0].isdigit() else 1, x[0].upper() if x[0].isalpha() else x[0])
            )
        )
        source_df_filtered = source_df[filtered_attrs]
        numerical_attrs = sorted(
            [col for col in source_df_filtered.select_dtypes(include=["number"]).columns],
            key=lambda x: (
                (0 if x[0].isdigit() else 1, x[0].upper() if x[0].isalpha() else x[0])
            )
        )
        categorical_attrs = [c for c in filtered_attrs if c not in numerical_attrs]    

        if set(source_df.columns) == set(target_df.columns):
            changed_attrs = [
                col for col in source_df.columns
                if not source_df[col].equals(target_df[col])
            ]       
        with col3:
            st.markdown("**Select the target attribute**")
            target_attr = st.radio(
                "none",
                changed_attrs,
                index = 0,
                label_visibility="hidden"
            )
            numerical_attrs = [c for c in numerical_attrs if c != target_attr]

        if target_attr is not None:
            ranked_condition_attrs, ranked_transformation_attrs = rank_attrs(
                    target_df=target_df,
                    categorical_attrs=categorical_attrs,
                    numerical_attrs=numerical_attrs,            
                    target_attr=target_attr)

            with col6:
                st.markdown("**Adjust system parameters**")
                # Slider 1: Max number for condition
                max_condition = st.slider(
                    "Max number for condition",
                    min_value=1,
                    max_value=len(ranked_condition_attrs),
                    value=3,
                    step=1
                )
                st.markdown("---", unsafe_allow_html=True)
                # Slider 2: Max number for transformation
                max_transformation = st.slider(
                    "Max number for transformation",
                    min_value=1,
                    max_value=len(ranked_transformation_attrs),
                    value=1,
                    step=1,
                    label_visibility="visible"
                )
                st.markdown("---", unsafe_allow_html=True)
                # Slider 3: Adjust value of weight parameter
                weight = st.slider(
                    "Adjust value of weight parameter (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )   

            with col4:
                st.markdown("**Select attributes for condition**")
                checked_items = {}
                for i, item in enumerate(ranked_condition_attrs):
                    checked_items[item] = st.checkbox(label=item, value=(i < 3), key=f"checkbox_{i}")

                selected_condition_attrs = []
                for item, checked in checked_items.items():
                    if checked:
                        selected_condition_attrs.append(item)

            with col5:
                st.markdown("**Select attributes for transformation**")
                checked_items = {}

                for i, item in enumerate(ranked_transformation_attrs):
                    checked_items[item] = st.checkbox(label=item, value=(i < 1), key=f"checkbox_{100+i}")

                selected_transformation_attrs = []
                for item, checked in checked_items.items():
                    if checked:
                        selected_transformation_attrs.append(item)

            if st.button("Summarize Difference"):
                st.session_state.show_summary_activated = True 
                st.session_state.recompute_summary = True
                st.session_state.clicked_row = -1         
            st.markdown("---")
    
        if st.session_state.show_summary_activated:
            populate_summary(source_df, 
                             target_df, 
                             target_attr, 
                             selected_condition_attrs, 
                             selected_transformation_attrs,
                             max_condition,
                             max_transformation,
                             weight)

setup_ui()






    



