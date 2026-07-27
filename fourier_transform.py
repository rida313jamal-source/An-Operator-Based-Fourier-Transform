import streamlit as st
import sympy as sp
import numpy as np
import math

def render_fourier_section():
    # ============================================================
    # HEADER & INTRODUCTION
    # ============================================================
    st.header("Part II: An Operator-Based Fourier Transform")

    st.markdown("""
    In this section we extend the operator-based Laplace framework to the oscillatory Fourier domain. The key structural point is that the bilateral Fourier transform naturally produces a symmetric pair of complex Laplace kernels rather than a single one-sided kernel.

    The regulator $e^{-\sigma|t|}$, with $\sigma > 0$, is not an artificial addition. It appears as the minimal symmetric damping required to make both half-axes integrable at once.

    This yields a regulated Fourier-Laplace operator transform, and the classical Fourier transform is recovered in the distributional limit $\sigma \to 0^+$.
    """)

    st.latex(r"s = \sigma + i\omega, \qquad \bar{s} = \sigma - i\omega, \qquad \sigma > 0")

    # ============================================================
    # DEFINITION
    # ============================================================
    st.subheader("Definition")

    st.latex(r"""
    \mathcal{F}_{\sigma}\{f\}(\omega)
    =
    \int_{-\infty}^{\infty} f(t)e^{-\sigma|t|}e^{-i\omega t}\,dt
    """)

    st.markdown("The planted Fourier operator acts on the symmetric kernel pair")
    st.latex(r"""
    \frac{1}{s},\qquad \frac{1}{\bar{s}}
    """)

    st.latex(r"""
    f(t)=\sum_{n=0}^{\infty} a_n t^n
    \quad\Longrightarrow\quad
    \mathcal{F}_{\sigma}\{f\}(\omega)
    =
    \sum_{n=0}^{\infty} a_n (-\partial_{\sigma})^n
    \left(
    \frac{1}{s}
    +
    (-1)^n\frac{1}{\bar{s}}
    \right)
    """)

    # ============================================================
    # DERIVATION EXPANDER
    # ============================================================
    with st.expander("Detailed derivation: from one-sided Laplace to regulated bilateral Fourier", expanded=True):
        st.markdown("**Step 1. Start from the formal bilateral Fourier integral**")
        st.latex(r"""
        \int_{-\infty}^{\infty} f(t)e^{-i\omega t}\,dt
        =
        \int_{0}^{\infty} f(t)e^{-i\omega t}\,dt
        +
        \int_{-\infty}^{0} f(t)e^{-i\omega t}\,dt
        """)

        st.markdown("**Step 2. Introduce the symmetric exponential regulator**")
        st.latex(r"w(t)=e^{-\sigma|t|},\qquad \sigma>0")
        st.latex(r"""
        \mathcal{F}_{\sigma}\{f\}(\omega)
        =
        \int_{-\infty}^{\infty} f(t)e^{-\sigma|t|}e^{-i\omega t}\,dt
        """)

        st.markdown("**Step 3. Split into positive and negative half-axes**")
        st.latex(r"""
        \mathcal{F}_{\sigma}\{f\}(\omega)
        =
        \int_{0}^{\infty} f(t)e^{-(\sigma+i\omega)t}\,dt
        +
        \int_{-\infty}^{0} f(t)e^{+\sigma t}e^{-i\omega t}\,dt
        """)

        st.markdown("**Step 4. Change variable on the negative half: \(u=-t\)**")
        st.latex(r"""
        \int_{-\infty}^{0} f(t)e^{+\sigma t}e^{-i\omega t}\,dt
        =
        \int_{0}^{\infty} f(-u)e^{-(\sigma-i\omega)u}\,du
        """)

        st.markdown("**Step 5. Identify the conjugate kernel pair**")
        st.latex(r"""
        s=\sigma+i\omega,\qquad \bar{s}=\sigma-i\omega
        """)
        st.latex(r"""
        \mathcal{F}_{\sigma}\{f\}(\omega)
        =
        \int_{0}^{\infty} f(t)e^{-st}\,dt
        +
        \int_{0}^{\infty} f(-t)e^{-\bar{s}t}\,dt
        """)

        st.markdown("**Step 6. Why the absolute value \(|t|\) appears**")
        st.latex(r"""
        e^{-\sigma|t|}
        =
        \begin{cases}
        e^{-\sigma t}, & t>0,\\
        e^{+\sigma t}, & t<0,
        \end{cases}
        """)

        st.markdown("""
        Hence the absolute value is a structural consequence of bilateral symmetric damping: it is exactly the choice that treats the positive and negative half-axes equally.
        """)

        st.markdown("**Step 7. Plant the monomial seed \(t^n\)**")
        st.latex(r"""
        \int_{0}^{\infty} t^n e^{-st}\,dt=\frac{\Gamma(n+1)}{s^{n+1}}
        \qquad\text{and}\qquad
        \int_{0}^{\infty} t^n e^{-\bar{s}t}\,dt=\frac{\Gamma(n+1)}{\bar{s}^{\,n+1}}
        """)

        st.latex(r"""
        \mathcal{F}_{\sigma}\{t^n\}(\omega)
        =
        \Gamma(n+1)\left(
        \frac{1}{s^{n+1}}
        +
        (-1)^n\frac{1}{\bar{s}^{\,n+1}}
        \right)
        """)

        st.markdown("**Step 8. Extend by Maclaurin linearity**")
        st.latex(r"""
        f(t)=\sum_{n=0}^{\infty} a_n t^n
        \quad\Longrightarrow\quad
        \mathcal{F}_{\sigma}\{f\}(\omega)
        =
        \sum_{n=0}^{\infty}
        a_n\Gamma(n+1)
        \left(
        \frac{1}{s^{n+1}}
        +
        (-1)^n\frac{1}{\bar{s}^{\,n+1}}
        \right)
        """)

        st.markdown("**Step 9. Distributional classical limit**")
        st.latex(r"""
        \lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{f\}(\omega)=\mathcal{F}\{f\}(\omega)
        """)

    st.divider()

    # ============================================================
    # CASE STUDIES
    # ============================================================
    st.subheader("Interactive Fourier Case Explorer")

    def get_fourier_cases():
        cases = {}

        # Case 1: f(t) = 1
        cases["Case 1: f(t) = 1"] = {
            "title": "Case 1: f(t) = 1",
            "function": r"f(t) = 1",
            "series": r"1 = \sum_{n \ge 0} a_n t^n,\qquad a_0=1,\; a_{n>0}=0.",
            "coefficients": r"a_0=1,\qquad a_{n>0}=0.",
            "rank": r"0",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{1\}(\omega)=\frac{1}{s}+\frac{1}{\bar s}",
                r"=\frac{1}{\sigma+i\omega}+\frac{1}{\sigma-i\omega}",
                r"=\frac{2\sigma}{\sigma^2+\omega^2}"
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{1\}(\omega)=\frac{2\sigma}{\sigma^2+\omega^2}",
            "classical_limit": r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{1\}(\omega)=2\pi\delta(\omega)",
            "extra_note": r"\lim_{\sigma\to0^+}\frac{1}{\pi}\frac{\sigma}{\sigma^2+\omega^2}=\delta(\omega)",
            "params": ["sigma", "omega"],
            "compute_result": lambda sigma_val, omega_val, **kwargs: (2 * sigma_val) / (sigma_val**2 + omega_val**2) if sigma_val > 0 else None,
        }

        # Case 2: f(t) = e^{-a|t|}
        cases["Case 2: f(t) = e^{-a|t|}"] = {
            "title": "Case 2: f(t) = e^{-a|t|}",
            "function": r"f(t) = e^{-a|t|}",
            "series": r"e^{-a|t|}=\sum_{n=0}^{\infty}\frac{(-a|t|)^n}{n!}",
            "coefficients": r"a_n=\frac{(-a)^n}{n!}",
            "rank": r"D^n",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\,\mathcal{F}_{\sigma}\{|t|^n\}(\omega)",
                r"\mathcal{F}_{\sigma}\{|t|^n\}(\omega)=\Gamma(n+1)\left[\frac{1}{s^{n+1}}+\frac{1}{\bar s^{\,n+1}}\right]",
                r"\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\Gamma(n+1)\left[\frac{1}{s^{n+1}}+\frac{1}{\bar s^{\,n+1}}\right]",
                r"\Gamma(n+1)=n!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\sum_{n=0}^{\infty}(-a)^n\left[\frac{1}{s^{n+1}}+\frac{1}{\bar s^{\,n+1}}\right]",
                r"=\frac{1}{s}\sum_{n=0}^{\infty}\left(-\frac{a}{s}\right)^n+\frac{1}{\bar s}\sum_{n=0}^{\infty}\left(-\frac{a}{\bar s}\right)^n",
                r"=\frac{1}{s+a}+\frac{1}{\bar s+a}",
                r"=\frac{1}{a+\sigma+i\omega}+\frac{1}{a+\sigma-i\omega}",
                r"=\frac{2(a+\sigma)}{(a+\sigma)^2+\omega^2}"
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\frac{2(a+\sigma)}{(a+\sigma)^2+\omega^2}",
            "classical_limit": r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\frac{2a}{a^2+\omega^2}",
            "extra_note": "",
            "params": ["sigma", "omega", "a"],
            "compute_result": lambda sigma_val, omega_val, a_val, **kwargs: (2 * (a_val + sigma_val)) / ((a_val + sigma_val)**2 + omega_val**2) if sigma_val > 0 and a_val > 0 else None,
        }

        # Case 3: f(t) = cos(t)
        cases["Case 3: f(t) = cos(t)"] = {
            "title": "Case 3: f(t) = cos(t)",
            "function": r"f(t) = \cos t",
            "series": r"\cos t=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}\,t^{2n}",
            "coefficients": r"a_{2n}=\frac{(-1)^n}{(2n)!}",
            "rank": r"2n",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}\,\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)",
                r"\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)=\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
                r"\Gamma(2n+1)=(2n)!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{\cos t\}(\omega)=\sum_{n=0}^{\infty}(-1)^n\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
                r"\sum_{n=0}^{\infty}\frac{(-1)^n}{s^{2n+1}}=\frac{1}{s}\sum_{n=0}^{\infty}\left(-\frac{1}{s^2}\right)^n=\frac{s}{s^2+1}",
                r"\sum_{n=0}^{\infty}\frac{(-1)^n}{\bar s^{\,2n+1}}=\frac{\bar s}{\bar s^{\,2}+1}",
                r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\frac{s}{s^2+1}+\frac{\bar s}{\bar s^{\,2}+1}",
                r"\frac{s}{s^2+1}=\frac12\left(\frac{1}{\sigma+i(\omega-1)}+\frac{1}{\sigma+i(\omega+1)}\right)",
                r"\frac{\bar s}{\bar s^{\,2}+1}=\frac12\left(\frac{1}{\sigma-i(\omega-1)}+\frac{1}{\sigma-i(\omega+1)}\right)",
                r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\frac{\sigma}{\sigma^2+(\omega-1)^2}+\frac{\sigma}{\sigma^2+(\omega+1)^2}"
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\frac{\sigma}{\sigma^2+(\omega-1)^2}+\frac{\sigma}{\sigma^2+(\omega+1)^2}",
            "classical_limit": r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\pi\,[\delta(\omega-1)+\delta(\omega+1)]",
            "extra_note": r"\text{Using }\lim_{\sigma\to0^+}\frac{\sigma}{(\omega-a)^2+\sigma^2}=\pi\,\delta(\omega-a)",
            "params": ["sigma", "omega"],
            "compute_result": lambda sigma_val, omega_val, **kwargs: (sigma_val / (sigma_val**2 + (omega_val - 1)**2)) + (sigma_val / (sigma_val**2 + (omega_val + 1)**2)) if sigma_val > 0 else None,
        }

        # Case 4: f(t) = sin(t)
        cases["Case 4: f(t) = sin(t)"] = {
            "title": "Case 4: f(t) = sin(t)",
            "function": r"f(t) = \sin t",
            "series": r"\sin t=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,t^{2n+1}",
            "coefficients": r"a_{2n+1}=\frac{(-1)^n}{(2n+1)!}",
            "rank": r"2n+1",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,\mathcal{F}_{\sigma}\{t^{2n+1}\}(\omega)",
                r"\mathcal{F}_{\sigma}\{t^{2n+1}\}(\omega)=\Gamma(2n+2)\left[\frac{1}{s^{2n+2}}-\frac{1}{\bar s^{\,2n+2}}\right]",
                r"\Gamma(2n+2)=(2n+1)!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{\sin t\}(\omega)=\sum_{n=0}^{\infty}(-1)^n\left[\frac{1}{s^{2n+2}}-\frac{1}{\bar s^{\,2n+2}}\right]",
                r"\sum_{n=0}^{\infty}\frac{(-1)^n}{s^{2n+2}}=\frac{1}{s^2+1},\qquad \sum_{n=0}^{\infty}\frac{(-1)^n}{\bar s^{\,2n+2}}=\frac{1}{\bar s^{\,2}+1}",
                r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{1}{s^2+1}-\frac{1}{\bar s^{\,2}+1}",
                r"\frac{1}{s^2+1}=\frac{1}{2i}\left[\frac{1}{\sigma+i(\omega-1)}-\frac{1}{\sigma+i(\omega+1)}\right]",
                r"\frac{1}{\bar s^{\,2}+1}=-\frac{1}{2i}\left[\frac{1}{\sigma-i(\omega-1)}-\frac{1}{\sigma-i(\omega+1)}\right]",
                r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{\sigma}{i}\left[\frac{1}{\sigma^2+(\omega-1)^2}-\frac{1}{\sigma^2+(\omega+1)^2}\right]"
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{\sigma}{i}\left[\frac{1}{\sigma^2+(\omega-1)^2}-\frac{1}{\sigma^2+(\omega+1)^2}\right]",
            "classical_limit": r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{\pi}{i}\,[\delta(\omega-1)-\delta(\omega+1)]",
            "extra_note": "",
            "params": ["sigma", "omega"],
            "compute_result": lambda sigma_val, omega_val, **kwargs: (sigma_val / 1j) * ((1 / (sigma_val**2 + (omega_val - 1)**2)) - (1 / (sigma_val**2 + (omega_val + 1)**2))) if sigma_val > 0 else None,
        }

        # Case 5: f(t) = e^{-a t^2}
        cases["Case 5: f(t) = e^{-a t^2}"] = {
            "title": "Case 5: f(t) = e^{-a t^2}",
            "function": r"f(t) = e^{-a t^2}",
            "series": r"e^{-a t^2}=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\,t^{2n}",
            "coefficients": r"a_{2n}=\frac{(-a)^n}{n!}",
            "rank": r"2n",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\,\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)",
                r"\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)=\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
                r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
                r"\Gamma(2n+1)=(2n)!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n(2n)!}{n!}\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
                r"\sum_{n=0}^{\infty}\frac{(-a)^n(2n)!}{n!}\frac{1}{z^{2n+1}}\ \text{resums to}\ \frac{\sqrt{\pi}}{2\sqrt{a}}\,e^{z^2/(4a)}\operatorname{erfc}\!\left(\frac{z}{2\sqrt a}\right)",
                r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\frac{\sqrt{\pi}}{2\sqrt a}\left[e^{s^2/(4a)}\operatorname{erfc}\!\left(\frac{s}{2\sqrt a}\right)+e^{\bar s^{\,2}/(4a)}\operatorname{erfc}\!\left(\frac{\bar s}{2\sqrt a}\right)\right]",
                r"\text{Taking } \sigma\to0^+ \text{ combines the two complementary-error terms and yields the Gaussian Fourier law.}"
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\frac{\sqrt{\pi}}{2\sqrt a}\left[e^{s^2/(4a)}\operatorname{erfc}\!\left(\frac{s}{2\sqrt a}\right)+e^{\bar s^{\,2}/(4a)}\operatorname{erfc}\!\left(\frac{\bar s}{2\sqrt a}\right)\right]",
            "classical_limit": r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sqrt{\frac{\pi}{a}}\,e^{-\omega^2/(4a)}",
            "extra_note": "",
            "params": ["sigma", "omega", "a"],
            "compute_result": lambda sigma_val, omega_val, a_val, **kwargs: math.sqrt(math.pi / a_val) * math.exp(-omega_val**2 / (4 * a_val)) if sigma_val > 0 and a_val > 0 else None,
        }

        # Case 6: sinc(t)
        cases["Case 6: sinc(t) = sin(t)/t"] = {
            "title": "Case 6: sinc(t) = sin(t)/t",
            "function": r"f(t) = \mathrm{sinc}(t) = \frac{\sin t}{t}",
            "series": r"\mathrm{sinc}(t)=\frac{\sin t}{t}=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,t^{2n}",
            "coefficients": r"a_{2n}=\frac{(-1)^n}{(2n+1)!}",
            "rank": r"2n",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)",
                r"\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)=\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar{s}^{\,2n+1}}\right]",
                r"\frac{\Gamma(2n+1)}{(2n+1)!}=\frac{(2n)!}{(2n+1)!}=\frac{1}{2n+1}",
                r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{2n+1}\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar{s}^{\,2n+1}}\right]",
                r"\sum_{n=0}^{\infty}\frac{(-1)^n z^{2n+1}}{2n+1}=\arctan(z)",
                r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\arctan\!\left(\frac{1}{s}\right)+\arctan\!\left(\frac{1}{\bar{s}}\right)",
                r"\text{Equivalently}",
                r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\arctan\!\left(\frac{1}{\sigma+i\omega}\right)+\arctan\!\left(\frac{1}{\sigma-i\omega}\right)",
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\arctan\!\left(\frac{1}{\sigma+i\omega}\right)+\arctan\!\left(\frac{1}{\sigma-i\omega}\right)",
            "classical_limit": r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\pi\,\mathbf{1}_{(|\omega|<1)},\qquad \mathcal{F}\{\mathrm{sinc}(t)\}(\pm1)=\frac{\pi}{2}",
            "extra_note": "",
            "params": ["sigma", "omega"],
            "compute_result": lambda sigma_val, omega_val, **kwargs: math.atan2(1, sigma_val + omega_val * 1j) + math.atan2(1, sigma_val - omega_val * 1j) if sigma_val > 0 else None,
        }

        # Case 7: delta(t)
        cases["Case 7: δ(t)"] = {
            "title": "Case 7: δ(t)",
            "function": r"f(t) = \delta(t)",
            "series": r"\delta(t)\ \text{is handled distributionally rather than by a Maclaurin series.}",
            "coefficients": r"\text{Mass concentrated at } t=0.",
            "rank": r"\text{distributional}",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{\delta(t)\}(\omega)=\int_{0}^{\infty}\delta(t)e^{-\sigma t}e^{-i\omega t}\,dt+\int_{0}^{\infty}\delta(-t)e^{-\sigma t}e^{+i\omega t}\,dt",
                r"\int_{0}^{\infty}\delta(t)\phi(t)\,dt=\frac12\phi(0),\qquad \delta(-t)=\delta(t)",
                r"\mathcal{F}_{\sigma}\{\delta(t)\}(\omega)=\frac12\cdot 1+\frac12\cdot 1=1"
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{\delta(t)\}(\omega)=1",
            "classical_limit": r"\mathcal{F}\{\delta(t)\}(\omega)=1",
            "extra_note": "",
            "params": ["sigma", "omega"],
            "compute_result": lambda sigma_val, omega_val, **kwargs: 1.0,
        }

        # Case 8: chi_{[-R,R]}(t)
        cases["Case 8: χ_{[-R,R]}(t)"] = {
            "title": "Case 8: χ_{[-R,R]}(t)",
            "function": r"f(t) = \chi_{[-R,R]}(t)",
            "series": r"\chi_{[-R,R]}(t)\ \text{is interval-supported and is best handled directly from the regulated bilateral form.}",
            "coefficients": r"\text{This case is finite-support rather than Maclaurin-planted.}",
            "rank": r"\text{finite interval case}",
            "plant_sum": [
                r"\mathcal{F}_{\sigma}\{\chi_{[-R,R]}\}(\omega)=\int_{-R}^{R}e^{-\sigma|t|}e^{-i\omega t}\,dt",
                r"=\int_{0}^{R}e^{-(\sigma+i\omega)t}\,dt+\int_{0}^{R}e^{-(\sigma-i\omega)t}\,dt",
                r"=\frac{1-e^{-sR}}{s}+\frac{1-e^{-\bar s R}}{\bar s}"
            ],
            "closed_form": r"\mathcal{F}_{\sigma}\{\chi_{[-R,R]}\}(\omega)=\frac{1-e^{-sR}}{s}+\frac{1-e^{-\bar s R}}{\bar s}",
            "classical_limit": r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\chi_{[-R,R]}\}(\omega)=\frac{1-e^{-i\omega R}}{i\omega}-\frac{1-e^{i\omega R}}{i\omega}=\frac{2\sin(\omega R)}{\omega}=2R\,\operatorname{sinc}(\omega R)",
            "extra_note": "",
            "params": ["sigma", "omega", "R"],
            "compute_result": lambda sigma_val, omega_val, R_val, **kwargs: (1 - math.exp(-(sigma_val + 1j * omega_val) * R_val)) / (sigma_val + 1j * omega_val) + (1 - math.exp(-(sigma_val - 1j * omega_val) * R_val)) / (sigma_val - 1j * omega_val) if sigma_val > 0 and R_val > 0 else None,
        }

        return cases

    cases = get_fourier_cases()
    selected_case = st.selectbox(
        "Choose a symbolic Fourier case",
        list(cases.keys()),
        index=0,
        key="fourier_detailed_case",
    )

    case = cases[selected_case]

    st.subheader(case["title"])

    st.markdown("**Function**")
    st.latex(case["function"])

    st.markdown("**Series**")
    st.latex(case["series"])

    st.markdown("**Coefficients**")
    st.latex(case["coefficients"])

    st.markdown("**Rank**")
    st.latex(case["rank"])

    st.markdown("**Plant & Sum**")
    for step in case["plant_sum"]:
        st.latex(step)

    st.markdown("**Closed Regulated Form**")
    st.latex(case["closed_form"])

    st.markdown("**Classical Limit**")
    st.latex(case["classical_limit"])

    if case.get("extra_note"):
        st.markdown("**Remark**")
        st.latex(case["extra_note"])

    # ============================================================
    # PARAMETER INPUTS AND NUMERICAL EVALUATION
    # ============================================================
    st.markdown("---")
    st.subheader("Numerical Evaluation")

    params = case.get("params", [])

    if "sigma" in params:
        sigma_val = st.number_input(
            "Enter value for $\\sigma$ (regulator)",
            value=0.1,
            step=0.05,
            format="%.2f",
            key=f"sigma_{selected_case[:10]}"
        )
    else:
        sigma_val = None

    if "omega" in params:
        omega_val = st.number_input(
            "Enter value for $\\omega$ (frequency)",
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"omega_{selected_case[:10]}"
        )
    else:
        omega_val = None

    if "a" in params:
        a_val = st.number_input(
            "Enter value for $a$",
            value=1.0,
            step=0.1,
            format="%.2f",
            key=f"a_fourier_{selected_case[:10]}"
        )
    else:
        a_val = None

    if "R" in params:
        R_val = st.number_input(
            "Enter value for $R$ (radius)",
            value=1.0,
            step=0.1,
            format="%.2f",
            key=f"R_fourier_{selected_case[:10]}"
        )
    else:
        R_val = None

    # Compute result
    if case.get("compute_result"):
        try:
            result = case["compute_result"](
                sigma_val=sigma_val if sigma_val is not None else 0.1,
                omega_val=omega_val if omega_val is not None else 0.0,
                a_val=a_val if a_val is not None else 1.0,
                R_val=R_val if R_val is not None else 1.0,
            )
            if result is not None:
                if isinstance(result, complex):
                    st.success(f"**Numerical Result:** $\\mathcal{{F}}_{{\\sigma}} \\{{f\\}}({omega_val if omega_val else 0.0})$ = {result.real:.6f} + {result.imag:.6f}i")
                else:
                    st.success(f"**Numerical Result:** $\\mathcal{{F}}_{{\\sigma}} \\{{f\\}}({omega_val if omega_val else 0.0})$ = {result:.6f}")
            else:
                st.warning("The entered values do not satisfy the convergence condition (σ > 0 required).")
        except Exception as e:
            st.error(f"Error computing result: {e}")

    st.divider()

    # ============================================================
    # TABLE: Summary of Fourier Results
    # ============================================================
    import streamlit as st

st.header("Table 2: Summary of Operator Fourier Transform Results")

# Fixed by using a Raw String (r""") and replacing internal markdown pipe characters '|' with '\vert'
st.markdown(r"""

| Function $f(t)$ | Regulated form $\mathcal{F}_\sigma\{f\}(\omega)$ | Classical limit $\lim_{\sigma\to0^+}$ |
|---|---|---|
| $1$ | $\frac{2\sigma}{\sigma^2+\omega^2}$ | $2\pi\delta(\omega)$ |
| $e^{-a\vert t\vert}$ | $\frac{2(a+\sigma)}{(a+\sigma)^2+\omega^2}$ | $\frac{2a}{a^2+\omega^2}$ |
| $\cos t$ | $\frac{\sigma}{\sigma^2+(\omega-1)^2}+\frac{\sigma}{\sigma^2+(\omega+1)^2}$ | $\pi[\delta(\omega-1)+\delta(\omega+1)]$ |
| $\sin t$ | $\frac{\sigma}{i}\left[\frac{1}{\sigma^2+(\omega-1)^2}-\frac{1}{\sigma^2+(\omega+1)^2}\right]$ | $\frac{\pi}{i}[\delta(\omega-1)-\delta(\omega+1)]$ |
| $e^{-a t^2}$ | $\frac{\sqrt{\pi}}{2\sqrt a}\left[e^{s^2/(4a)}\operatorname{erfc}\left(\frac{s}{2\sqrt a}\right)+e^{\bar s^2/(4a)}\operatorname{erfc}\left(\frac{\bar s}{2\sqrt a}\right)\right]$ | $\sqrt{\frac{\pi}{a}}e^{-\omega^2/(4a)}$ |
| $\mathrm{sinc}(t)$ | $\arctan\left(\frac{1}{\sigma+i\omega}\right)+\arctan\left(\frac{1}{\sigma-i\omega}\right)$ | $\pi\mathbf{1}_{(\vert\omega\vert<1)}$ |
| $\delta(t)$ | $1$ | $1$ |
| $\chi_{[-R,R]}(t)$ | $\frac{1-e^{-sR}}{s}+\frac{1-e^{-\bar s R}}{\bar s}$ | $\frac{2\sin(\omega R)}{\omega}$ |
""")

st.markdown(r"""
$s = \sigma + i\omega,\qquad \bar{s} = \sigma - i\omega$
""")


if __name__ == "__main__":
    render_fourier_section()
