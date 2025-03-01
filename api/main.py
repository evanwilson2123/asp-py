from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
import requests
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

##############################
# PDF GENERATION CODE
##############################

class FormData(BaseModel):
    firstName: str
    lastName: str
    height: str
    weight: float
    age: int
    primarySport: str
    currentTrainingReg: str
    goals: str
    primaryPosition: str
    hopeToGain: str
    injuryHistory: str
    coachingStyle: str
    daysTraining: int
    priorSC: bool
    # Mobility Assessment
    overHeadSquat: int
    trunkStability: int
    sidePlank: int
    spinalFlexion: int
    activeLegRaise: int
    goodMorning: int
    lungeOverhead: int
    lateralTrunkTilt: int
    mobilityNotes: str
    # Hitting Mechanics Breakdown
    weighShift: int
    torsoRot: int
    pelvisLoad: int
    forwardMove: int
    hipShoulder: int
    upperRot: int
    lowerRot: int
    frontArm: int
    shoulderConn: int
    barrelExt: int
    batShoulderAng: int
    hittingNotes: str
    # Pitching Mechanics Breakdown
    startingPos: int
    legLiftInitWeightShift: int
    engageGlute: int
    pushBackLeg: int
    vertShinAngViR: int
    stayHeel: int
    driveDirection: int
    outDriveEarly: int
    latVertGround: int
    backKneeDrive: int
    hipClear: int
    rotDown: int
    movesIndependent: int
    excessiveRot: int
    earlyTorsoRot: int
    torsoNotSegment: int
    bowFlexBow: int
    scapularDig: int
    reflexivePecFire: int
    armSlotTorsoRot: int
    rotPerpSpine: int
    excessiveTilt: int
    throwUpHill: int
    armSwingCapMom: int
    overlyPronOrSup: int
    overlyFlexOrExtWrist: int
    elbowInLine: int
    lateEarlyFlipUp: int
    elbowFlexionHundred: int
    fullScapRetractAbduct: int
    armDrag: int
    limitedLayback: int
    elbowPushForward: int
    straightElbowNeutral: int
    armWorksInd: int
    earlySup: int
    workOppGlove: int
    retractAbductLanding: int
    rotatesIntoPlane: int
    leaks: int
    frontFootContact: int
    pawback: int
    kneeStabTran: int
    kneeStabFron: int
    forearmPron: int
    shoulderIntern: int
    scapRelease: int
    thoracicFlex: int
    noViolentRecoil: int
    overallTempo: int
    overallRhythm: int
    propTimedIntent: int
    cervPos: int
    pitchingNotes: str

@app.post("/gen-pdf")
def gen_pdf(data: FormData):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch
    )
    elements = []
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = styles["Heading2"]
    normal_style = styles["Normal"]
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=subtitle_style,
        spaceAfter=10,
        textColor=colors.white,
        backColor=colors.darkblue,
        alignment=1,
        fontSize=14,
        leading=16
    )
    elements.append(Paragraph(f"Athlete Performance Report For {data.firstName} {data.lastName}", title_style))
    elements.append(Spacer(1, 12))
    def add_section(title, fields):
        elements.append(Paragraph(title, section_title_style))
        elements.append(Spacer(1, 6))
        table_data = []
        for key, value in fields.items():
            cell_value = Paragraph(str(value), normal_style) if key.lower().endswith("notes") else str(value)
            table_data.append([key.replace("_", " ").title(), cell_value])
        table = Table(table_data, colWidths=[200, 250])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))
    add_section("General Information", {
        "Height": data.height,
        "Weight": f"{data.weight} lbs",
        "Age": data.age,
        "Primary Sport": data.primarySport,
        "Primary Position": data.primaryPosition,
        "Current Training Regimen": data.currentTrainingReg,
        "Goals": data.goals,
        "Hope to Gain": data.hopeToGain,
        "Injury History": data.injuryHistory,
        "Coaching Style": data.coachingStyle,
        "Days Training per Week": data.daysTraining,
        "Prior Strength & Conditioning": "Yes" if data.priorSC else "No",
    })
    add_section("Mobility Assessment", {
        "Overhead Squat": data.overHeadSquat,
        "Trunk Stability": data.trunkStability,
        "Side Plank": data.sidePlank,
        "Spinal Flexion": data.spinalFlexion,
        "Active Leg Raise": data.activeLegRaise,
        "Good Morning": data.goodMorning,
        "Lunge Overhead": data.lungeOverhead,
        "Lateral Trunk Tilt": data.lateralTrunkTilt,
        "Mobility Notes": data.mobilityNotes
    })
    add_section("Hitting Mechanics Breakdown", {
        "Weigh Shift": data.weighShift,
        "Torso Rotation": data.torsoRot,
        "Pelvis Load": data.pelvisLoad,
        "Forward Move": data.forwardMove,
        "Hip Shoulder Separation": data.hipShoulder,
        "Upper Rotation": data.upperRot,
        "Lower Rotation": data.lowerRot,
        "Front Arm Movement": data.frontArm,
        "Shoulder Connection": data.shoulderConn,
        "Barrel Extension": data.barrelExt,
        "Bat Shoulder Angle": data.batShoulderAng,
        "Hitting Notes": data.hittingNotes
    })
    add_section("Pitching Mechanics Breakdown", {
        "Starting Position": data.startingPos,
        "Leg Lift + Initial Weight Shift": data.legLiftInitWeightShift,
        "Glute Engagement": data.engageGlute,
        "Back Leg Drive": data.pushBackLeg,
        "Vertical Shin Angle": data.vertShinAngViR,
        "Stay On Heel": data.stayHeel,
        "Drive Direction": data.driveDirection,
        "Out of Drive Early": data.outDriveEarly,
        "Lateral Vertical Ground": data.latVertGround,
        "Back Knee Drive": data.backKneeDrive,
        "Hip Clearance": data.hipClear,
        "Rotation Down": data.rotDown,
        "Independent Movements": data.movesIndependent,
        "Excessive Rotation": data.excessiveRot,
        "Early Torso Rotation": data.earlyTorsoRot,
        "Torso Not Segmenting": data.torsoNotSegment,
        "Bow Flex Bow": data.bowFlexBow,
        "Scapular Dig": data.scapularDig,
        "Reflexive Pectoral Fire": data.reflexivePecFire,
        "Arm Slot & Torso Rotation": data.armSlotTorsoRot,
        "Rotation Perpendicular to Spine": data.rotPerpSpine,
        "Excessive Tilt": data.excessiveTilt,
        "Throw Uphill": data.throwUpHill,
        "Arm Swing & Cap Momentum": data.armSwingCapMom,
        "Overly Pronated or Supinated": data.overlyPronOrSup,
        "Overly Flexed or Extended Wrist": data.overlyFlexOrExtWrist,
        "Elbow In Line": data.elbowInLine,
        "Late or Early Flip Up": data.lateEarlyFlipUp,
        "Elbow Flexion to 100 Degrees": data.elbowFlexionHundred,
        "Full Scapular Retraction & Abduction": data.fullScapRetractAbduct,
        "Arm Drag": data.armDrag,
        "Limited Layback": data.limitedLayback,
        "Elbow Pushing Forward": data.elbowPushForward,
        "Straight or Neutral Elbow": data.straightElbowNeutral,
        "Arm Works Independently": data.armWorksInd,
        "Early Supination": data.earlySup,
        "Works Opposite to Glove": data.workOppGlove,
        "Retract & Abduct at Landing": data.retractAbductLanding,
        "Rotates Into Plane": data.rotatesIntoPlane,
        "Leaks Energy": data.leaks,
        "Front Foot Contact": data.frontFootContact,
        "Pawback Mechanics": data.pawback,
        "Knee Stability Transition": data.kneeStabTran,
        "Knee Stability Front Side": data.kneeStabFron,
        "Forearm Pronation": data.forearmPron,
        "Shoulder Internal Rotation": data.shoulderIntern,
        "Scapular Release": data.scapRelease,
        "Thoracic Flexion": data.thoracicFlex,
        "No Violent Recoil": data.noViolentRecoil,
        "Overall Tempo": data.overallTempo,
        "Overall Rhythm": data.overallRhythm,
        "Properly Timed Intent": data.propTimedIntent,
        "Cervical Position": data.cervPos,
        "Pitching Notes": data.pitchingNotes,
    })
    doc.build(elements)
    buffer.seek(0)
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=athlete_report.pdf"}
    )

##############################
# MODEL LOADING FROM VERCEL BLOB
##############################

def load_model_from_blob(url: str):
    """Download a .sav file from Vercel Blob and unpickle it."""
    resp = requests.get(url)
    resp.raise_for_status()
    return pickle.loads(resp.content)

# Replace these placeholders with your actual public Blob URLs.
RFC_MODELFB_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/rfc_modelfb.sav"
RFC_MODELCB_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/rfc_modelcb.sav"
RFC_MODELSL_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/rfc_modelsl.sav"
RFC_MODELCH_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/rfc_modelch.sav"

XGB_MODELFB_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/xgb_modelfb.sav"
XGB_MODELCB_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/xgb_modelcb.sav"
XGB_MODELSL_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/xgb_modelsl.sav"
XGB_MODELCH_URL = "https://iqpjsciijbncme4r.public.blob.vercel-storage.com/models/xgb_modelch.sav"

# Load models at startup (cold start)
rf_models = {
    "Fastball": load_model_from_blob(RFC_MODELFB_URL),
    "Sinker":   load_model_from_blob(RFC_MODELFB_URL),
    "Curveball": load_model_from_blob(RFC_MODELCB_URL),
    "Slider":   load_model_from_blob(RFC_MODELSL_URL),
    "Cutter":   load_model_from_blob(RFC_MODELSL_URL),
    "ChangeUp": load_model_from_blob(RFC_MODELCH_URL)
}

xgb_models = {
    "Fastball": load_model_from_blob(XGB_MODELFB_URL),
    "Sinker":   load_model_from_blob(XGB_MODELFB_URL),
    "Curveball": load_model_from_blob(XGB_MODELCB_URL),
    "Slider":   load_model_from_blob(XGB_MODELSL_URL),
    "Cutter":   load_model_from_blob(XGB_MODELSL_URL),
    "ChangeUp": load_model_from_blob(XGB_MODELCH_URL)
}

##############################
# STUFF+ CALCULATOR ENDPOINT
##############################

class PitchData(BaseModel):
    Pitch_Type: str
    RelSpeed: float
    SpinRate: float
    RelHeight: float
    ABS_RelSide: float
    Extension: float
    ABS_Horizontal: float = None
    InducedVertBreak: float = None
    differential_break: float = None

def calculate_stuff_plus(row: pd.Series):
    pitch_type = row['Pitch_Type']
    if pitch_type in rf_models:
        rf_model = rf_models[pitch_type]
        xgb_model = xgb_models[pitch_type]
        if pitch_type in ['Fastball', 'Sinker']:
            features = ['RelSpeed', 'SpinRate', 'differential_break', 'RelHeight', 'ABS_RelSide', 'Extension']
        else:
            features = ['RelSpeed', 'SpinRate', 'ABS_Horizontal', 'RelHeight', 'ABS_RelSide', 'Extension', 'InducedVertBreak']
        try:
            row_features = row[features].values.reshape(1, -1)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Missing or invalid features: {e}")
        xWhiff_rf = rf_model.predict_proba(row_features)[0][1]
        xWhiff_xg = xgb_model.predict_proba(row_features)[0][1]
        xWhiff = (xWhiff_rf + xWhiff_xg) / 2
        if pitch_type in ['Fastball', 'Sinker']:
            return (xWhiff / 0.18206374469443068) * 100
        elif pitch_type in ['Curveball', 'KnuckleCurve']:
            return (xWhiff / 0.30139757759674063) * 100
        elif pitch_type in ['Slider', 'Cutter']:
            return (xWhiff / 0.32823183402173944) * 100
        elif pitch_type in ['ChangeUp']:
            return (xWhiff / 0.32612872148563093) * 100
    else:
        raise HTTPException(status_code=400, detail="Invalid pitch type")

@app.post("/calculate-stuff")
def calculate_stuff_endpoint(pitch: PitchData):
    data = pitch.dict()
    if data["Pitch_Type"] in ["Fastball", "Sinker"]:
        if data.get("differential_break") is None:
            if data.get("ABS_Horizontal") is None or data.get("InducedVertBreak") is None:
                raise HTTPException(
                    status_code=400,
                    detail="Missing ABS_Horizontal or InducedVertBreak to compute differential_break"
                )
            data["differential_break"] = abs(data["InducedVertBreak"] - data["ABS_Horizontal"])
    row = pd.Series(data)
    result = calculate_stuff_plus(row)
    return {"stuff_plus": result}

@app.get("/")
def health_check():
    return {"Server": "Healthy"}
