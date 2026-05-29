#include "inflated.h"

static inline double stable_pc(double eta0, double eta1) {
    double e0 = std::exp(-eta0), e1 = std::exp(-eta1);
    return (e1 - e0) / ((1.0 + e0) * (1.0 + e1));
}

double AgreementPhi::inflated::obs_loglik(
    double Y,
    double ALPHA,
    double PHI,
    double K0,
    double K1,
    double &DALPHA,
    double &NEG_D2ALPHA,
    int GRADFLAG
) {
    double eta0 = ALPHA - K0;
    double eta1 = ALPHA - K1;
    double L0  = link::mu(eta0);
    double L1  = link::mu(eta1);
    double mu  = link::mu(ALPHA);
    double v0  = L0 * (1.0 - L0);
    double v1  = L1 * (1.0 - L1);
    double vmu = mu * (1.0 - mu);

    if (Y == 0.0) {
        double ll = std::log1p(-L0);
        if (GRADFLAG > 0) DALPHA      += -L0;
        if (GRADFLAG > 1) NEG_D2ALPHA += v0;
        return ll;
    }

    if (Y == 1.0) {
        double ll = std::log(L1);
        if (GRADFLAG > 0) DALPHA      += 1.0 - L1;
        if (GRADFLAG > 1) NEG_D2ALPHA += v1;
        return ll;
    }

    double pc = stable_pc(eta0, eta1);
    if (pc <= 0.0) return -std::numeric_limits<double>::infinity();

    double a = mu * PHI;
    double b = (1.0 - mu) * PHI;
    if (!(a > 1e-15 && b > 1e-15)) return -std::numeric_limits<double>::infinity();

    double ll_beta = std::lgamma(PHI) - std::lgamma(a) - std::lgamma(b)
                   + (a - 1.0) * std::log(Y) + (b - 1.0) * std::log(1.0 - Y);
    if (!std::isfinite(ll_beta)) return -std::numeric_limits<double>::infinity();
    double ll = std::log(pc) + ll_beta;

    if (GRADFLAG > 0) {
        double y_star  = std::log(Y / (1.0 - Y));
        double mu_star = boost::math::digamma(a) - boost::math::digamma(b);
        double sc      = 1.0 - L0 - L1;
        DALPHA += sc + PHI * (y_star - mu_star) * vmu;

        if (GRADFLAG > 1) {
            double umu    = vmu * (1.0 - 2.0 * mu);
            double trig_a = boost::math::trigamma(a);
            double trig_b = boost::math::trigamma(b);
            double neg_d2 = (v0 + v1)
                          + PHI * PHI * (trig_a + trig_b) * vmu * vmu
                          - PHI * (y_star - mu_star) * umu;
            NEG_D2ALPHA += neg_d2;
        }
    }

    return ll;
}

double AgreementPhi::inflated::E0_dalpha0dalpha1(
    double ALPHA0, double PHI0, double K0_0, double K1_0,
    double ALPHA1, double PHI1, double K0_1, double K1_1
) {
    double eta0_0 = ALPHA0 - K0_0, eta1_0 = ALPHA0 - K1_0;
    double L0_0 = link::mu(eta0_0);
    double L1_0 = link::mu(eta1_0);
    double mu0  = link::mu(ALPHA0);
    double vmu0 = mu0 * (1.0 - mu0);
    double pc_0 = stable_pc(eta0_0, eta1_0);

    double eta0_1 = ALPHA1 - K0_1, eta1_1 = ALPHA1 - K1_1;
    double L0_1 = link::mu(eta0_1);
    double L1_1 = link::mu(eta1_1);
    double mu1  = link::mu(ALPHA1);
    double vmu1 = mu1 * (1.0 - mu1);
    double pc_1 = stable_pc(eta0_1, eta1_1);

    double s0_0 = -L0_0;
    double s1_0 = 1.0 - L1_0;
    double s0_1 = -L0_1;
    double s1_1 = 1.0 - L1_1;

    double p0_0 = 1.0 - L0_0;
    double p1_0 = L1_0;

    double result = p0_0 * s0_0 * s0_1 + p1_0 * s1_0 * s1_1;

    if (pc_0 > 0.0 && pc_1 > 0.0) {
        double sc_0 = 1.0 - L0_0 - L1_0;
        double sc_1 = 1.0 - L0_1 - L1_1;
        double C  = PHI1 * vmu1;
        double C0 = PHI0 * vmu0;
        double A0 = boost::math::digamma(mu0 * PHI0) - boost::math::digamma((1.0 - mu0) * PHI0);
        double A1 = boost::math::digamma(mu1 * PHI1) - boost::math::digamma((1.0 - mu1) * PHI1);
        double B0 = boost::math::trigamma(mu0 * PHI0) + boost::math::trigamma((1.0 - mu0) * PHI0);
        double interior = sc_0 * sc_1 + sc_0 * C * (A0 - A1) + C * C0 * B0;
        result += pc_0 * interior;
    }

    return result;
}

double AgreementPhi::inflated::brent_alpha(
    const std::vector<double>& Y,
    const std::vector<int>& ITEM_OBS,
    double ALPHA_START,
    double PHI,
    double K0,
    double K1,
    double LOWER,
    double UPPER,
    int MAX_ITER
) {
    auto neg_ll = [&](double alpha) {
        double dalpha = 0.0, neg_d2alpha = 0.0;
        double nll = 0.0;
        for (int idx : ITEM_OBS) {
            nll -= obs_loglik(Y[idx], alpha, PHI, K0, K1, dalpha, neg_d2alpha, 0);
        }
        return nll;
    };

    boost::uintmax_t maxit = static_cast<boost::uintmax_t>(MAX_ITER);
    const int digits = std::numeric_limits<double>::digits;
    auto res = boost::math::tools::brent_find_minima(neg_ll, LOWER, UPPER, digits, maxit);
    return res.first;
}
