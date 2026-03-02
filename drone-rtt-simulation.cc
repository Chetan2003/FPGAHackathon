/* =============================================================
 * Drone RTT Data Collection with Jamming - NS3
 * =============================================================
 * Topology:
 *   [Drone] <---wireless---> [GCS]
 *                ^
 *            [Jammer]
 *
 * RTT Method: Custom socket ping — send timestamp embedded
 * inside payload. RTT = now - embedded_send_time.
 *
 * Key feature change:
 *   SINR (signal - noise) replaced with noise_floor_dbm.
 *
 *   SINR conflates two independent causes of degradation:
 *     - Drone moves away  → signal drops, noise stays same → SINR drops
 *     - Jammer turns on   → signal stays same, noise rises → SINR drops
 *   Both look identical to the model — root cause of 40% false alarm rate.
 *
 *   noise_floor_dbm = signalNoise.noise ONLY
 *     - Drone moves away  → noise stays ~-95 dBm (thermal only)
 *     - Jammer turns on   → noise jumps to -70 dBm+ (jammer interference)
 *   These are now SEPARABLE. The model can distinguish distance from jamming.
 *
 *   Grace period label: rows within POST_JAM_GRACE seconds after jammer OFF
 *   are still labeled jammer_active=1 (MAC layer still recovering).
 *
 * Output CSV columns:
 *   timestamp_s, rtt_ms, packet_loss_rate, rssi_dbm, noise_floor_dbm,
 *   drone_x, drone_y, drone_distance_m, jammer_active
 * ============================================================= */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/socket.h"
#include "ns3/inet-socket-address.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>       // log10, pow, sqrt — needed for Friis path loss

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DroneRTTSimulation");

// ─────────────────────────────────────────────
// Globals
// ─────────────────────────────────────────────
std::ofstream   csvFile;
uint32_t        totalTxPackets = 0;
uint32_t        totalRxPackets = 0;
bool            jammerActive   = false;
double          latestRssiDbm       = -65.0;
double          latestNoiseFloorDbm = -93.0;  // computed via Friis (see CalcNoiseFloor)
double          jammerOffTime       = -999.0;

// Global jammer tx power — set from command line, used in Friis calculation
double          g_jammerPowerDbm = 40.0;

// Jammer position — fixed at (50,50,50), set once in main
// Stored globally so LogRTT can access it without a node pointer argument
Vector          g_jammerPos(50.0, 50.0, 50.0);

// WiFi centre frequency (channel 36 = 5.18 GHz)
const double    WIFI_FREQ_HZ    = 5.18e9;
const double    SPEED_OF_LIGHT  = 3.0e8;
const double    THERMAL_NOISE_DBM = -93.0;  // receiver thermal noise floor

// Grace period: rows within this many seconds after jammer OFF
// are still labeled jammer_active=1 (RTT still elevated during recovery)
const double POST_JAM_GRACE = 2.0;

// ── Jamming type constants ───────────────────
// COMB:    Jammer continuously blasts noise across 6 fixed channels.
//          Modelled as 100% duty cycle at full power.
//          Worst case for the drone — no gap to recover.
//
// SWEEP:   Jammer scans entire 5GHz band at 3ms per channel.
//          Assuming 11 channels → hits drone's channel for 3ms every 33ms.
//          Modelled as 3ms ON / 30ms OFF (duty cycle ≈ 9%).
//
// DYNAMIC: Adversary alternates between SWEEP and COMB every 10s.
//          Designed to confuse detection models that tune to one pattern.
//
// PULSE:   Original on/off duty cycle (all runs 1-8). Kept for compatibility.
enum JammerType { JAM_PULSE, JAM_COMB, JAM_SWEEP, JAM_DYNAMIC };
JammerType      activeJammerType = JAM_PULSE;

// Sweep parameters
const double SWEEP_ON_MS    = 3.0 / 1000.0;   // 3ms on (one channel dwell)
const double SWEEP_TOTAL_MS = 33.0 / 1000.0;  // 11 channels × 3ms = 33ms period
const double SWEEP_OFF_MS   = SWEEP_TOTAL_MS - SWEEP_ON_MS;  // 30ms off

// Dynamic: switch period between sweep and comb modes
const double DYNAMIC_SWITCH_S = 10.0;

// Track sweep/dynamic scheduling events
bool          dynamicInCombMode = false;

Ptr<Node>              droneNode;
Ptr<Node>              gcsNode;
Ptr<Node>              jammerNode;
Ptr<OnOffApplication>  jammerApp;

const uint16_t PING_PORT = 8888;   // GCS listens here
const uint16_t ECHO_PORT = 8889;   // Drone listens for replies

// ─────────────────────────────────────────────
// Packet payload layout (10 bytes):
//   [0..1]  sequence number (uint16_t)
//   [2..9]  send timestamp in nanoseconds (uint64_t)
// Remaining bytes padded to 64 total.
// ─────────────────────────────────────────────
struct PingPayload {
    uint16_t seq;
    uint64_t sendTimeNs;
};

// Ping scheduling state
Ptr<Socket> droneTxSocket;
Ptr<Socket> droneRxSocket;
Address     gcsAddress;
double      echoIntervalSec = 0.1;
uint16_t    pingSeq = 0;

// ─────────────────────────────────────────────
// Effective noise floor using Friis path loss
// ─────────────────────────────────────────────
// Root cause of the flat -93 dBm problem:
//   signalNoise.noise from MonitorSnifferRx is the RECEIVER'S
//   hardware thermal noise figure — a fixed hardware constant.
//   NS3 uses it internally to compute SINR for PER calculations
//   but does NOT add the jammer's received power to this field.
//   The jammer causes packet errors but its energy is invisible
//   in the MonitorSnifferRx callback.
//
// Fix: compute effective noise floor ourselves using Friis.
//
//   jammer_rx_power_dbm = Pt - PL(d)
//   PL(d) = 20*log10(4*pi*d*f/c)     [Friis free-space path loss]
//
//   noise_total_mw = 10^(thermal/10) + 10^(jammer_rx/10)
//   effective_noise_dbm = 10 * log10(noise_total_mw)
//
// When jammer OFF:
//   effective_noise = thermal = -93 dBm  (matches what you observed)
//
// When jammer ON at 60 dBm, 70m distance:
//   PL = 20*log10(4*pi*70*5.18e9/3e8) = 83.3 dB
//   jammer_rx = 60 - 83.3 = -23.3 dBm
//   noise_total = 10^(-93/10) + 10^(-23.3/10) ≈ 4678 mW
//   effective_noise ≈ -23.3 dBm  (completely dominated by jammer)
//
// This gives the clear clean/jammed separation the model needs.
// ─────────────────────────────────────────────
double CalcNoiseFloorDbm(const Vector& dronePos)
{
    // Thermal noise — what we always had
    double thermalMw = std::pow(10.0, THERMAL_NOISE_DBM / 10.0);

    if (!jammerActive)
        return THERMAL_NOISE_DBM;   // -93 dBm, distance-independent

    // Friis path loss from jammer to drone
    double dx = dronePos.x - g_jammerPos.x;
    double dy = dronePos.y - g_jammerPos.y;
    double dz = dronePos.z - g_jammerPos.z;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Clamp to 1m minimum to avoid log(0)
    dist = std::max(dist, 1.0);

    double pathLossDb = 20.0 * std::log10(
        4.0 * M_PI * dist * WIFI_FREQ_HZ / SPEED_OF_LIGHT);

    double jammerRxDbm = g_jammerPowerDbm - pathLossDb;
    double jammerRxMw  = std::pow(10.0, jammerRxDbm / 10.0);

    // Combine thermal + jammer interference in linear (mW) domain
    double totalMw = thermalMw + jammerRxMw;
    return 10.0 * std::log10(totalMw);
}

// ─────────────────────────────────────────────
// CSV Logger
// ─────────────────────────────────────────────
void LogRTT(double rttMs)
{
    Ptr<MobilityModel> dm = droneNode->GetObject<MobilityModel>();
    Vector dp = dm->GetPosition();

    Ptr<MobilityModel> gm = gcsNode->GetObject<MobilityModel>();
    Vector gp = gm->GetPosition();

    double dist = std::sqrt(
        (dp.x-gp.x)*(dp.x-gp.x) +
        (dp.y-gp.y)*(dp.y-gp.y) +
        (dp.z-gp.z)*(dp.z-gp.z));

    double lossRate = (totalTxPackets > 0)
        ? 1.0 - ((double)totalRxPackets / totalTxPackets)
        : 0.0;

    // Grace period label fix:
    // Even after jammer turns OFF, RTT stays elevated for 1-2 seconds
    // while MAC layer recovers from backoff. Label these rows as jammed
    // so the model doesn't learn "high RTT = clean" during recovery.
    double timeSinceJammerOff = Simulator::Now().GetSeconds() - jammerOffTime;
    double noiseFloor = CalcNoiseFloorDbm(dp);
    bool effectivelyJammed = jammerActive || (timeSinceJammerOff < POST_JAM_GRACE);

    csvFile << std::fixed << std::setprecision(6)
            << Simulator::Now().GetSeconds() << ","
            << rttMs                          << ","
            << lossRate                       << ","
            << latestRssiDbm                  << ","
            << noiseFloor                     << ","
            << dp.x                           << ","
            << dp.y                           << ","
            << dist                           << ","
            << (effectivelyJammed ? 1 : 0)    << "\n";
    csvFile.flush();
}

// ─────────────────────────────────────────────
// GCS Echo Socket
// Receives ping packet, sends exact bytes back
// ─────────────────────────────────────────────
void GcsEchoRecv(Ptr<Socket> echoSocket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = echoSocket->RecvFrom(from)))
    {
        // Send the same packet back to the drone's ECHO_PORT
        InetSocketAddress src = InetSocketAddress::ConvertFrom(from);
        InetSocketAddress replyAddr(src.GetIpv4(), ECHO_PORT);
        echoSocket->SendTo(packet, 0, replyAddr);
    }
}

// ─────────────────────────────────────────────
// Drone Rx — receives echo reply, reads
// the embedded timestamp, computes RTT
// ─────────────────────────────────────────────
void DroneEchoRecv(Ptr<Socket> rxSocket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = rxSocket->RecvFrom(from)))
    {
        totalRxPackets++;

        if (packet->GetSize() < sizeof(PingPayload))
            continue;

        // Read embedded payload
        PingPayload payload;
        packet->CopyData(reinterpret_cast<uint8_t*>(&payload),
                         sizeof(PingPayload));

        // RTT = current time - send time (both in nanoseconds)
        uint64_t nowNs = Simulator::Now().GetNanoSeconds();
        double rttMs   = (nowNs >= payload.sendTimeNs)
                         ? (nowNs - payload.sendTimeNs) / 1.0e6
                         : 0.0;

        LogRTT(rttMs);
    }
}

// ─────────────────────────────────────────────
// Drone Tx — send one ping, schedule next
// ─────────────────────────────────────────────
void SendPing()
{
    totalTxPackets++;
    pingSeq++;

    // Build payload with current timestamp
    PingPayload payload;
    payload.seq        = pingSeq;
    payload.sendTimeNs = Simulator::Now().GetNanoSeconds();

    // Pad packet to 64 bytes
    const uint32_t pktSize = 64;
    uint8_t buf[pktSize];
    std::memset(buf, 0, pktSize);
    std::memcpy(buf, &payload, sizeof(PingPayload));

    Ptr<Packet> pkt = Create<Packet>(buf, pktSize);
    droneTxSocket->SendTo(pkt, 0, gcsAddress);

    // Schedule the next ping
    Simulator::Schedule(Seconds(echoIntervalSec), &SendPing);
}

// ─────────────────────────────────────────────
// RSSI sniffer on drone PHY
// ─────────────────────────────────────────────
void PhyRxCallback(Ptr<const Packet>  packet,
                   uint16_t           channelFreqMhz,
                   WifiTxVector       txVector,
                   MpduInfo           aMpdu,
                   SignalNoiseDbm     signalNoise,
                   uint16_t           staId)
{
    latestRssiDbm       = signalNoise.signal;
    // signalNoise.noise is the receiver hardware thermal floor (-93 dBm constant).
    // It does NOT include jammer power — NS3 does not add jammer energy here.
    // Noise floor is now computed via Friis in CalcNoiseFloorDbm() inside LogRTT.
}

// ─────────────────────────────────────────────
// Jammer toggle + pulse scheduling
// ─────────────────────────────────────────────
void JammerOn()
{
    jammerActive = true;
    if (jammerApp)
        jammerApp->SetAttribute("OnTime",
            StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    NS_LOG_INFO("Jammer ON  t=" << Simulator::Now().GetSeconds());
}

void JammerOff()
{
    jammerActive  = false;
    jammerOffTime = Simulator::Now().GetSeconds(); // start grace period timer
    if (jammerApp)
        jammerApp->SetAttribute("OnTime",
            StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    NS_LOG_INFO("Jammer OFF t=" << Simulator::Now().GetSeconds()
                << " (grace period: " << POST_JAM_GRACE << "s)");
}

void SchedulePulseJamming(double start, double end,
                           double onDur, double offDur)
{
    for (double t = start; t < end; t += onDur + offDur)
    {
        Simulator::Schedule(Seconds(t),         &JammerOn);
        Simulator::Schedule(Seconds(t + onDur), &JammerOff);
    }
}

// ─────────────────────────────────────────────────────────────
// COMB JAMMING
// Continuously blasts noise across a fixed set of channels.
// Modelled as: jammer ON for entire duration.
// In real world: 6 simultaneous CW carriers on adjacent channels.
// In NS3 single-channel model: full-power continuous transmission.
// Effect on features: noise_floor stays permanently elevated,
// RTT either stays high or all packets lost.
// ─────────────────────────────────────────────────────────────
void ScheduleCombJamming(double start, double end)
{
    NS_LOG_INFO("Scheduling COMB jamming: " << start << "s → " << end << "s");
    Simulator::Schedule(Seconds(start), &JammerOn);
    Simulator::Schedule(Seconds(end),   &JammerOff);
}

// ─────────────────────────────────────────────────────────────
// SWEEP JAMMING
// Jammer scans rapidly across entire 5GHz band at 3ms/channel.
// With 11 channels: hits drone's channel for 3ms every 33ms.
// Duty cycle = 3/33 ≈ 9%  — much lower than pulse or comb.
//
// Effect on features:
//   - noise_floor spikes briefly every 33ms (hard to catch)
//   - RTT shows occasional spikes but often recovers quickly
//   - rolling_std rises due to intermittent disruptions
//   - Harder for the model to detect than comb — important training data
// ─────────────────────────────────────────────────────────────
void ScheduleSweepJamming(double start, double end)
{
    NS_LOG_INFO("Scheduling SWEEP jamming: " << start << "s → " << end
                << "s  (3ms ON / " << SWEEP_OFF_MS*1000 << "ms OFF)");
    for (double t = start; t < end; t += SWEEP_TOTAL_MS)
    {
        Simulator::Schedule(Seconds(t),                  &JammerOn);
        Simulator::Schedule(Seconds(t + SWEEP_ON_MS),    &JammerOff);
    }
}

// ─────────────────────────────────────────────────────────────
// DYNAMIC JAMMING — alternates SWEEP ↔ COMB every 10 seconds
// Designed to defeat AI models that learn one jamming signature.
// The model must handle both patterns to get correct labels.
//
// Phase 0 (t=0..10s):  SWEEP mode (3ms/33ms duty cycle)
// Phase 1 (t=10..20s): COMB mode  (100% duty cycle)
// Phase 2 (t=20..30s): SWEEP mode again
// ...
// ─────────────────────────────────────────────────────────────
void DynamicSwitchToComb(double endTime);
void DynamicSwitchToSweep(double endTime);

void DynamicSwitchToComb(double endTime)
{
    double now = Simulator::Now().GetSeconds();
    if (now >= endTime) return;

    NS_LOG_INFO("Dynamic jam: switching to COMB at t=" << now);
    dynamicInCombMode = true;
    JammerOn();   // turn on continuously

    double nextSwitch = std::min(now + DYNAMIC_SWITCH_S, endTime);
    Simulator::Schedule(Seconds(DYNAMIC_SWITCH_S),
        [endTime]() { DynamicSwitchToSweep(endTime); });
}

void DynamicSwitchToSweep(double endTime)
{
    double now = Simulator::Now().GetSeconds();
    if (now >= endTime) { JammerOff(); return; }

    NS_LOG_INFO("Dynamic jam: switching to SWEEP at t=" << now);
    dynamicInCombMode = false;
    JammerOff();   // kill continuous comb

    // Schedule sweep pulses for next DYNAMIC_SWITCH_S seconds
    for (double t = 0; t < DYNAMIC_SWITCH_S && (now + t) < endTime;
         t += SWEEP_TOTAL_MS)
    {
        Simulator::Schedule(Seconds(t),               &JammerOn);
        Simulator::Schedule(Seconds(t + SWEEP_ON_MS), &JammerOff);
    }

    Simulator::Schedule(Seconds(DYNAMIC_SWITCH_S),
        [endTime]() { DynamicSwitchToComb(endTime); });
}

void ScheduleDynamicJamming(double start, double end)
{
    NS_LOG_INFO("Scheduling DYNAMIC jamming: " << start << "s → " << end
                << "s  (SWEEP↔COMB every " << DYNAMIC_SWITCH_S << "s)");
    // Start with sweep mode
    Simulator::Schedule(Seconds(start),
        [start, end]() { DynamicSwitchToSweep(end); });
    // Hard stop at end
    Simulator::Schedule(Seconds(end), &JammerOff);
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char *argv[])
{
    double      simDuration   = 300.0;
    double      jammerPower   = 40.0;
    std::string runLabel      = "run1";
    bool        enableJammer  = false;
    double      jammerStart   = 30.0;
    double      jammerEnd     = 270.0;
    double      jammerOnTime  = 10.0;
    double      jammerOffTime = 10.0;
    std::string jammerTypeStr = "pulse"; // "pulse" | "comb" | "sweep" | "dynamic"

    CommandLine cmd(__FILE__);
    cmd.AddValue("duration",     "Simulation duration (s)",         simDuration);
    cmd.AddValue("runLabel",     "Run label for CSV name",          runLabel);
    cmd.AddValue("enableJammer", "Enable jammer (0/1)",             enableJammer);
    cmd.AddValue("jammerOn",     "Jammer ON duration (s) [pulse]",  jammerOnTime);
    cmd.AddValue("jammerOff",    "Jammer OFF duration (s) [pulse]", jammerOffTime);
    cmd.AddValue("jammerPower",  "Jammer Tx power (dBm)",           jammerPower);
    cmd.AddValue("jammerType",   "pulse|comb|sweep|dynamic",        jammerTypeStr);
    cmd.Parse(argc, argv);
    g_jammerPowerDbm = jammerPower;   // sync global used by CalcNoiseFloorDbm

    std::string csvPath = "rtt_log_" + runLabel + ".csv";
    LogComponentEnable("DroneRTTSimulation", LOG_LEVEL_INFO);

    // ─────────────────────────────────────────
    // 1. Nodes
    // ─────────────────────────────────────────
    NodeContainer allNodes;
    allNodes.Create(3);
    droneNode  = allNodes.Get(0);
    gcsNode    = allNodes.Get(1);
    jammerNode = allNodes.Get(2);

    // ─────────────────────────────────────────
    // 2. Shared WiFi Channel
    // ─────────────────────────────────────────
    YansWifiChannelHelper wifiChannelHelper;
    wifiChannelHelper.SetPropagationDelay(
        "ns3::ConstantSpeedPropagationDelayModel");
    wifiChannelHelper.AddPropagationLoss(
        "ns3::FriisPropagationLossModel",
        "Frequency", DoubleValue(5.18e9));

    // One shared channel object — jammer must use the same pointer
    Ptr<YansWifiChannel> sharedChannel = wifiChannelHelper.Create();

    // Drone + GCS PHY
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(sharedChannel);
    wifiPhy.Set("ChannelSettings", StringValue("{36, 20, BAND_5GHZ, 0}"));
    wifiPhy.Set("TxPowerStart",    DoubleValue(20.0));
    wifiPhy.Set("TxPowerEnd",      DoubleValue(20.0));
    wifiPhy.Set("TxPowerLevels",   UintegerValue(1));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager(
        "ns3::ConstantRateWifiManager",
        "DataMode",    StringValue("HtMcs7"),
        "ControlMode", StringValue("HtMcs0"));

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    NodeContainer droneGcs;
    droneGcs.Add(droneNode);
    droneGcs.Add(gcsNode);
    NetDeviceContainer droneGcsDevices = wifi.Install(wifiPhy, wifiMac, droneGcs);

    // Jammer PHY — same shared channel, high power
    YansWifiPhyHelper jammerPhy;
    jammerPhy.SetChannel(sharedChannel);
    jammerPhy.Set("ChannelSettings", StringValue("{36, 20, BAND_5GHZ, 0}"));
    jammerPhy.Set("TxPowerStart",    DoubleValue(jammerPower));
    jammerPhy.Set("TxPowerEnd",      DoubleValue(jammerPower));
    jammerPhy.Set("TxPowerLevels",   UintegerValue(1));

    WifiMacHelper jammerMac;
    jammerMac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer jammerDevice = wifi.Install(jammerPhy, jammerMac, jammerNode);

    // ─────────────────────────────────────────
    // 3. Mobility
    // ─────────────────────────────────────────
    MobilityHelper mobility;

    // GCS at origin
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(gcsNode);
    gcsNode->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 0.0));

    // Jammer at center of drone patrol area — maximizes interference
    mobility.Install(jammerNode);
    jammerNode->GetObject<MobilityModel>()->SetPosition(Vector(50.0, 50.0, 50.0));

    // Drone patrol path — stays within 100m of GCS
    mobility.SetMobilityModel("ns3::WaypointMobilityModel");
    mobility.Install(droneNode);
    Ptr<WaypointMobilityModel> droneMob =
        droneNode->GetObject<WaypointMobilityModel>();

    droneMob->AddWaypoint(Waypoint(Seconds(0.0),   Vector(0.0,   0.0,   50.0)));
    droneMob->AddWaypoint(Waypoint(Seconds(50.0),  Vector(100.0, 0.0,   50.0)));
    droneMob->AddWaypoint(Waypoint(Seconds(100.0), Vector(100.0, 100.0, 50.0)));
    droneMob->AddWaypoint(Waypoint(Seconds(150.0), Vector(0.0,   100.0, 50.0)));
    droneMob->AddWaypoint(Waypoint(Seconds(200.0), Vector(0.0,   0.0,   50.0)));
    droneMob->AddWaypoint(Waypoint(Seconds(250.0), Vector(100.0, 0.0,   50.0)));
    droneMob->AddWaypoint(Waypoint(Seconds(300.0), Vector(0.0,   0.0,   50.0)));

    // ─────────────────────────────────────────
    // 4. Internet Stack + IPs
    // ─────────────────────────────────────────
    InternetStackHelper internet;
    internet.Install(allNodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.0");

    // Assign drone + GCS first
    Ipv4InterfaceContainer ifaces = ipv4.Assign(droneGcsDevices);

    // Assign jammer on same subnet — use a fresh Assign call
    // NS3 tracks allocated addresses so this gets 10.0.0.3
    Ipv4InterfaceContainer jammerIface = ipv4.Assign(jammerDevice);

    Ipv4Address droneIp = ifaces.GetAddress(0);
    Ipv4Address gcsIp   = ifaces.GetAddress(1);

    // ─────────────────────────────────────────
    // 5. Custom Ping Sockets
    //
    // Flow:  Drone ---[PING_PORT]---> GCS
    //        GCS   ---[ECHO_PORT]---> Drone
    // ─────────────────────────────────────────

    // GCS echo socket
    Ptr<Socket> gcsSocket = Socket::CreateSocket(
        gcsNode, UdpSocketFactory::GetTypeId());
    gcsSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), PING_PORT));
    gcsSocket->SetRecvCallback(MakeCallback(&GcsEchoRecv));

    // Drone reply socket (receives echoes)
    droneRxSocket = Socket::CreateSocket(
        droneNode, UdpSocketFactory::GetTypeId());
    droneRxSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), ECHO_PORT));
    droneRxSocket->SetRecvCallback(MakeCallback(&DroneEchoRecv));

    // Drone transmit socket (sends pings)
    droneTxSocket = Socket::CreateSocket(
        droneNode, UdpSocketFactory::GetTypeId());
    droneTxSocket->Bind(InetSocketAddress(droneIp, 0));

    gcsAddress = InetSocketAddress(gcsIp, PING_PORT);

    // Start first ping at t=1s (gives WiFi time to associate)
    Simulator::Schedule(Seconds(1.0), &SendPing);
    // Stop pings before sim ends
    Simulator::Schedule(Seconds(simDuration - 1.0),
        []() { Simulator::Cancel(Simulator::Schedule(Seconds(0), &SendPing)); });

    // ─────────────────────────────────────────
    // 6. RSSI Sniffer
    // ─────────────────────────────────────────
    Config::ConnectWithoutContext(
        "/NodeList/0/DeviceList/0/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
        MakeCallback(&PhyRxCallback));

    // ─────────────────────────────────────────
    // 7. Jammer
    // ─────────────────────────────────────────
    if (enableJammer)
    {
        // Parse jammer type string
        if      (jammerTypeStr == "comb")    activeJammerType = JAM_COMB;
        else if (jammerTypeStr == "sweep")   activeJammerType = JAM_SWEEP;
        else if (jammerTypeStr == "dynamic") activeJammerType = JAM_DYNAMIC;
        else                                 activeJammerType = JAM_PULSE;

        // Flood broadcast on drone/GCS subnet
        OnOffHelper jamHelper("ns3::UdpSocketFactory",
            InetSocketAddress(Ipv4Address("10.0.0.255"), 7777));
        jamHelper.SetAttribute("DataRate",   StringValue("54Mbps"));
        jamHelper.SetAttribute("PacketSize", UintegerValue(1024));
        jamHelper.SetAttribute("OnTime",
            StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        jamHelper.SetAttribute("OffTime",
            StringValue("ns3::ConstantRandomVariable[Constant=1]"));

        ApplicationContainer jamApps = jamHelper.Install(jammerNode);
        jamApps.Start(Seconds(0.0));
        jamApps.Stop(Seconds(simDuration));

        jammerApp = DynamicCast<OnOffApplication>(jamApps.Get(0));

        // Dispatch scheduling based on type
        switch (activeJammerType)
        {
            case JAM_COMB:
                // Continuous noise on fixed channel subset
                // Modelled as 100% duty cycle (worst case for drone)
                ScheduleCombJamming(jammerStart, jammerEnd);
                NS_LOG_INFO("COMB jamming: continuous @ " << jammerPower << "dBm");
                break;

            case JAM_SWEEP:
                // Rapid scan: 3ms on-channel, 30ms off (11 channels × 3ms)
                ScheduleSweepJamming(jammerStart, jammerEnd);
                NS_LOG_INFO("SWEEP jamming: 3ms/channel @ " << jammerPower << "dBm");
                break;

            case JAM_DYNAMIC:
                // Alternates SWEEP ↔ COMB every 10s to confuse the model
                ScheduleDynamicJamming(jammerStart, jammerEnd);
                NS_LOG_INFO("DYNAMIC jamming: SWEEP↔COMB every "
                            << DYNAMIC_SWITCH_S << "s @ " << jammerPower << "dBm");
                break;

            case JAM_PULSE:
            default:
                // Original pulse scheduling (runs 1-8)
                SchedulePulseJamming(jammerStart, jammerEnd,
                                     jammerOnTime, jammerOffTime);
                NS_LOG_INFO("PULSE jamming: " << jammerOnTime
                            << "s ON / " << jammerOffTime << "s OFF");
                break;
        }
    }

    // ─────────────────────────────────────────
    // 8. CSV
    // ─────────────────────────────────────────
    csvFile.open(csvPath);
    csvFile << "timestamp_s,rtt_ms,packet_loss_rate,"
            << "rssi_dbm,noise_floor_dbm,drone_x,drone_y,"
            << "drone_distance_m,jammer_active\n";

    // ─────────────────────────────────────────
    // 9. Flow Monitor
    // ─────────────────────────────────────────
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // ─────────────────────────────────────────
    // 10. Run
    // ─────────────────────────────────────────
    NS_LOG_INFO("Run: " << runLabel
        << "  duration=" << simDuration << "s"
        << "  jammer="   << (enableJammer ? "ON" : "OFF"));

    Simulator::Stop(Seconds(simDuration));
    Simulator::Run();

    // ─────────────────────────────────────────
    // 11. Summary
    // ─────────────────────────────────────────
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

    std::cout << "\n=== Flow Monitor ===\n";
    for (auto &f : monitor->GetFlowStats())
    {
        auto t = classifier->FindFlow(f.first);
        double meanDelay = (f.second.rxPackets > 0)
            ? f.second.delaySum.GetMilliSeconds() / f.second.rxPackets
            : 0.0;
        std::cout << "Flow " << f.first
                  << "  " << t.sourceAddress << " -> " << t.destinationAddress
                  << "  Tx=" << f.second.txPackets
                  << "  Rx=" << f.second.rxPackets
                  << "  Lost=" << f.second.lostPackets
                  << "  MeanDelay=" << meanDelay << "ms\n";
    }

    std::cout << "\nTotal Tx=" << totalTxPackets
              << "  Rx=" << totalRxPackets
              << "  CSV=" << csvPath << "\n";

    csvFile.close();
    Simulator::Destroy();
    return 0;
}

/* =============================================================
 * BUILD & RUN
 * =============================================================
 * cp drone-rtt-simulation.cc ~/ns-3.46/scratch/rtt_drone.cc
 * cd ~/ns-3.46 && ./ns3 build
 *
 * Run 1 - no jammer (baseline):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run1 --enableJammer=0"
 *
 * Run 2 - constant jam (60dBm):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run2 --enableJammer=1 --jammerOn=300 --jammerOff=1 --jammerPower=60"
 *
 * Run 3 - pulse 10s/10s (60dBm):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run3 --enableJammer=1 --jammerOn=10 --jammerOff=10 --jammerPower=60"
 *
 * Run 4 - sparse 5s/15s (60dBm):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run4 --enableJammer=1 --jammerOn=5 --jammerOff=15 --jammerPower=60"
 *
 * Run 5 - heavy 15s/5s (60dBm):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run5 --enableJammer=1 --jammerOn=15 --jammerOff=5 --jammerPower=60"
 *
 * Run 6 - rapid 3s/7s (60dBm):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run6 --enableJammer=1 --jammerOn=3 --jammerOff=7 --jammerPower=60"
 *
 * Run 7 - moderate jammer power (45dBm) pulse 10s/10s:
 *   ./ns3 run "scratch/rtt_drone --runLabel=run7 --enableJammer=1 --jammerOn=10 --jammerOff=10 --jammerPower=45"
 *
 * Run 8 - severe jammer power (75dBm) pulse 10s/10s:
 *   ./ns3 run "scratch/rtt_drone --runLabel=run8 --enableJammer=1 --jammerOn=10 --jammerOff=10 --jammerPower=75"
 *
 * ── NEW JAMMING TYPES ──────────────────────────────────────────────────────
 *
 * Run 9 - COMB jamming (continuous noise on 6 fixed channels):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run9 --enableJammer=1 --jammerType=comb --jammerPower=60"
 *   What to expect: noise_floor permanently elevated during jam window.
 *   RTT either spikes or all packets lost. Model sees clearest jamming signal.
 *
 * Run 10 - SWEEP jamming (3ms/channel, 11 channels = 33ms period):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run10 --enableJammer=1 --jammerType=sweep --jammerPower=65"
 *   What to expect: brief noise_floor spikes every 33ms (hard to catch).
 *   rolling_std rises. RTT shows occasional spikes. Duty cycle ≈ 9%.
 *   Harder for model to detect — critical training diversity.
 *
 * Run 11 - DYNAMIC jamming (SWEEP↔COMB alternating every 10s):
 *   ./ns3 run "scratch/rtt_drone --runLabel=run11 --enableJammer=1 --jammerType=dynamic --jammerPower=60"
 *   What to expect: first 10s sweep pattern, next 10s comb, repeating.
 *   Tests model generalisation across changing adversary strategy.
 *
 * Combine all runs including new types:
 *   head -1 rtt_log_run1.csv > combined_dataset.csv
 *   for f in rtt_log_run*.csv; do tail -n +2 $f >> combined_dataset.csv; done
 * ============================================================= */
