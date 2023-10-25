import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FixedLocator, FixedFormatter

plt.rcParams["text.usetex"] = True


def SolitonVelocity():
    return waveNumber**2 + 6 * integrationConst * np.ones(np.size(waveNumber))

def PhaseArg(x, t):
    return waveNumber * (x * np.ones(np.size(waveNumber)) - SolitonVelocity() * t - phaseShift)

def PhaseArgGhost(x, t, waveNumber_, phaseShift_):
    velocity = waveNumber_**2 + 6 * integrationConst
    return waveNumber_ * (x - velocity * t - phaseShift_)

def CouplingA(i, j):
    return (waveNumber[i] - waveNumber[j])**2 / (waveNumber[i] + waveNumber[j])**2

def CouplingAA(i, j):
    return (waveNumber[i] - waveNumber[j])**2 / (waveNumber[i] + waveNumber[j])

def CouplingAAA(i, j):
    return (waveNumber[i] - waveNumber[j])**2

def Numerator1(x, t):
    numerator = 0
    sumPhaseArgs = 0

    for i, k in enumerate(waveNumber):
        numerator += k**2 * np.exp(PhaseArg(x, t)[i])
        sumPhaseArgs += PhaseArg(x, t)[i]

    numerator += CouplingAAA(0, 1) * np.exp(sumPhaseArgs)

    return numerator

def Numerator2(x, t):
    numerator = 0
    sumPhaseArgs = 0

    for i, k in enumerate(waveNumber):
        numerator += k * np.exp(PhaseArg(x, t)[i])
        sumPhaseArgs += PhaseArg(x, t)[i]

    numerator += CouplingAA(0, 1) * np.exp(sumPhaseArgs)

    return numerator

def Denominator(x, t):
    denominator = 1
    sumPhaseArgs = 0

    for i, _ in enumerate(waveNumber):
        denominator += np.exp(PhaseArg(x, t)[i])
        sumPhaseArgs += PhaseArg(x, t)[i]

    denominator += CouplingA(0, 1) * np.exp(sumPhaseArgs)

    return denominator

def MultiSoliton(x, t):
    return 2 * Numerator1(x, t) / Denominator(x, t) \
        - 2 * (Numerator2(x, t) / Denominator(x, t))**2 \
        + integrationConst

def SingleSolitonGhost(x, t, waveNumber_, phaseShift_):
    expArg = np.exp(PhaseArgGhost(x, t, waveNumber_, phaseShift_))
    return 2 * waveNumber_**2 * expArg / (1 + expArg)**2 + integrationConst

def SolitonMaxTravelTime():
    travelTimes = []
        
    for i, v in enumerate(SolitonVelocity()):
        if abs(v) < 1.0E-6: # i.e. velocity is zero
            continue
        elif abs(v) < 0.5: 
            # exclude snail-pace velocity, it would make the travel time exceedingly long
            continue
        elif v > 0:
            travelDistance = xMax - phaseShift[i]
        elif v < 0:
            travelDistance = phaseShift[i] - xMin

        travelTimes.append(travelDistance / abs(v))

    maxTravelTime = max(travelTimes)

    return round(maxTravelTime)


fig, ax1 = plt.subplots(figsize=(16, 9))
ax2 = ax1.twiny()

    
integrationConst = -2/3
waveNumber = np.array([2.1, 1.8])
phaseShift = np.array([-15, 10])
# TRIAL AND ERROR CORRECTIONS TO INITIAL POSITIONS (PEAK LOCATIONS) OF GHOST SOLITONS:
ghostInitialPositionCorrection = np.array([0.0, 2.9])

xMin = -20
xMax = -xMin

plotPointsPerUnitLength = 10
plotPoints = round(plotPointsPerUnitLength * (xMax - xMin))
xIncrement = (xMax - xMin) / (plotPoints - 1)
print(f"x interval = {xMax - xMin}, plotPointsPerUnitLength = {plotPointsPerUnitLength}, plotPoints = {plotPoints}, xIncrement = {xIncrement:.2E}")

x = np.linspace(xMin, xMax, plotPoints)

yMin = integrationConst
yMax = np.max([MultiSoliton(xx, 0) for xx in x])
yExtra = 0.2
ax1.set_xlim([xMin, xMax])
ax1.set_ylim([yMin - yExtra, yMax + yExtra])
ax2.set_xlim([xMin, xMax])

tStart = 0
tStop = SolitonMaxTravelTime()
timePointsPerUnitTime = 2
timePoints = 1 + round(timePointsPerUnitTime * (tStop - tStart))
tIncrement = (tStop - tStart) / (timePoints - 1)

print(f"t inverval = {tStop - tStart}, timePointsPerUnitTime = {timePointsPerUnitTime}, timePoints = {timePoints}, tIncrement = {tIncrement}")

for i, time in enumerate(np.linspace(tStart, tStop, timePoints)):

    y = [MultiSoliton(xx, time) for xx in x]
    y0 = [SingleSolitonGhost(xx, time, waveNumber[0], phaseShift[0] + ghostInitialPositionCorrection[0]) for xx in x]
    y1 = [SingleSolitonGhost(xx, time, waveNumber[1], phaseShift[1] + ghostInitialPositionCorrection[1]) for xx in x]


    latexExpression1 = r"""\parbox{20cm}{\center 
    Two-Soliton Solution of KdV Equation:\\[1ex]
    $u_t(x, t) + 6 u(x, t) u_x(x, t) + u_{xxx}(x, t) = 0$
    }""".replace('\n',' ')
    # \\[1ex]Bounce-exchange collision: solitons do not superpose

    latexExpression2 = r"""\parbox{20cm}{
    $\eta_i(x, t) = k_i \bigl( x - \underbrace{(k_i^2 - 6 a)}_{c_i \mathrm{\ (velocity)}} t - \delta_i \bigr), \quad i \in \{ 0, 1 \}, \quad t = %.1f$
    \\[1ex] $u( \{ \eta_i(x, t) \} ) = \cdots + a\quad$ (a closed-form expression)
    \\[1ex] $k_i = \{ %.2f, %.2f \}, \quad \delta_i = \{ %.2f, %.2f \}$
    \\[1ex] $c_i = \{ %.2f, %.2f \}, \quad a = %.3f$ (integration const.)
    }""".replace('\n',' ') % (time, waveNumber[0], waveNumber[1], phaseShift[0], phaseShift[1], SolitonVelocity()[0], SolitonVelocity()[1], integrationConst)

    ax1.set_title(latexExpression1, fontsize=25, pad=70)
    ax1.set_xlabel("$x$", fontsize=20)
    ax1.set_ylabel("$y$", fontsize=20)


    ax1.plot(x, y, linestyle="solid", linewidth=3, color="brown", label=latexExpression2)
    ax1.plot(x, y0, linestyle="dotted", linewidth=2, color="orange", label="``Ghost'': an individual would-be soliton in the absence of collision")
    ax1.plot(x, y1, linestyle="dotted", linewidth=2, color="orange")


    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_major_formatter('{x:.1f}')

    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_formatter('{x:.1f}') # must be "x" not "y"

    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    # ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(3))

    x2Ticks = FixedLocator(phaseShift)
    x2Labels = FixedFormatter([r"$\delta_0$", r"$\delta_1$"])
    ax2.xaxis.set_major_locator(x2Ticks)
    ax2.xaxis.set_major_formatter(x2Labels)

    ax1.tick_params(which='major', labelsize=15, length=10, width=2, direction='inout', top=False, right=False)
    ax1.tick_params(which='minor', length=5, width=2, direction='in', top=False, right=False)
    ax2.tick_params(which="major", labelsize=16, length=10, width=2, direction='inout', bottom=False, top=True, right=False)

    ax1.grid(which="major", linestyle='-.', color="grey", linewidth=1.5)
    ax1.grid(which="minor", linestyle=':', color="grey", linewidth=1)
    ax2.grid(which="major", linestyle='--', color="grey", linewidth=1.5)
    # ax1.legend(loc="best", fontsize=20).get_frame().set_alpha(1)
    ax1.legend(bbox_to_anchor=(0.8, -0.15), fontsize=20)
    plt.tight_layout()
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    plt.savefig("Plots/Animate/TwoSolitonCollision/TwoSolitonCollision" + str(i) + ".jpeg")

    ax1.clear()
    ax2.clear()
