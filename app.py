import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from src.search import search_voice
from src.feature_extractor import extract_features

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Voice Similarity Search",
    layout="centered"
)

# ====================================
# TITLE
# ====================================

st.title("🎤 Voice Similarity Search")

st.write(
    "Upload một file giọng nữ để tìm 5 giọng giống nhất trong dataset"
)

# ====================================
# FILE UPLOAD
# ====================================

uploaded_file = st.file_uploader(
    "Upload WAV file",
    type=["wav", "mp3"]
)

# ====================================
# MAIN SEARCH
# ====================================

if uploaded_file is not None:

    # play uploaded audio
    st.audio(uploaded_file)

    # save temp file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    st.info("Đang tìm kiếm giọng nói tương đồng...")

    try:

        # ====================================
        # EXTRACT QUERY FEATURES
        # ====================================

        query_features = extract_features(temp_path)

        # ====================================
        # SHOW QUERY FEATURES
        # ====================================

        st.header("📊 Đặc trưng âm thanh đầu vào")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Pitch (F0 Mean)",
                f"{query_features['pitch']['f0_mean']:.2f}"
            )

            st.metric(
                "Spectral Centroid",
                f"{query_features['spectral']['centroid']:.2f}"
            )

        with col2:

            st.metric(
                "MFCC Dimensions",
                len(query_features["mfcc_mean"])
            )

            st.metric(
                "Embedding Dimensions",
                len(query_features["speaker_embedding"])
            )

        # ====================================
        # EMBEDDING VISUALIZATION
        # ====================================

        st.subheader("🎯 Speaker Embedding Visualization")

        emb_preview = query_features[
            "speaker_embedding"
        ][:50]

        emb_df = pd.DataFrame({

            "Embedding Value": emb_preview

        })

        st.line_chart(emb_df)

        # ====================================
        # SEARCH
        # ====================================

        results = search_voice(temp_path)

        st.success("Tìm kiếm hoàn tất")

        st.header("🔎 Top 5 kết quả giống nhất")

        similarity_scores = []
        filenames = []

        # ====================================
        # SHOW RESULTS
        # ====================================

        for idx, r in enumerate(results, start=1):

            similarity = float(r["similarity"])

            similarity_percent = similarity * 100

            filenames.append(r["filename"])
            similarity_scores.append(similarity)

            st.markdown(f"# {idx}")

            st.write(f"📁 File: {r['filename']}")

            # similarity %
            st.metric(
                "Similarity Score",
                f"{similarity_percent:.2f}%"
            )

            # similarity bar
            st.progress(
                min(max(similarity, 0.0), 1.0)
            )

            # raw cosine score
            st.code(
                f"Cosine Similarity: {similarity:.6f}"
            )

            # interpretation
            if similarity >= 0.85:

                st.success(
                    "Rất giống giọng nói đầu vào"
                )

            elif similarity >= 0.70:

                st.info(
                    "Khá giống"
                )

            else:

                st.warning(
                    "Độ tương đồng thấp"
                )

            # ====================================
            # RESULT AUDIO
            # ====================================

            audio_path = os.path.join(
                "processed",
                r["filename"]
            )

            if os.path.exists(audio_path):

                st.audio(audio_path)

                # ====================================
                # RESULT FEATURES
                # ====================================

                result_features = extract_features(
                    audio_path
                )

                st.subheader(
                    "📌 Đặc trưng âm thanh"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Pitch",
                        f"{result_features['pitch']['f0_mean']:.2f}"
                    )

                    st.metric(
                        "Spectral",
                        f"{result_features['spectral']['centroid']:.2f}"
                    )

                with c2:

                    st.metric(
                        "MFCC Size",
                        len(result_features["mfcc_mean"])
                    )

                    st.metric(
                        "Embedding Size",
                        len(
                            result_features[
                                "speaker_embedding"
                            ]
                        )
                    )

                # embedding preview
                result_emb = result_features[
                    "speaker_embedding"
                ][:50]

                result_df = pd.DataFrame({

                    "Embedding": result_emb

                })

                st.line_chart(result_df)

            st.divider()

        # ====================================
        # DATAFRAME
        # ====================================

        st.header("📋 Similarity Table")

        df = pd.DataFrame({

            "Filename": filenames,

            "Cosine Similarity": similarity_scores

        })

        st.dataframe(df)

        # ====================================
        # BAR CHART
        # ====================================

        st.header("📈 Similarity Visualization")

        chart_df = pd.DataFrame(

            {
                "Similarity": similarity_scores
            },

            index=filenames
        )

        st.bar_chart(chart_df)

    except Exception as e:

        st.error(str(e))