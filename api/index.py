from fastapi import FastAPI, Response
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO

app = FastAPI()


# Define the Data Model (Matches TypeScript `FormData`)
class FormData(BaseModel):
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


@app.post("/gen-pdf")
def gen_pdf(data: FormData):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=1 * inch, bottomMargin=1 * inch)
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = styles["Heading2"]
    normal_style = styles["Normal"]

    # Custom Styles
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=subtitle_style,
        spaceAfter=10,
        textColor=colors.white,
        backColor=colors.darkblue,
        alignment=1,  # Centered
        fontSize=14,
        leading=16
    )

    # Title
    elements.append(Paragraph("Athlete Performance Report", title_style))
    elements.append(Spacer(1, 12))

    # Function to format sections
    def add_section(title, fields):
        elements.append(Paragraph(title, section_title_style))
        elements.append(Spacer(1, 6))

        table_data = [[key.replace("_", " ").title(), str(value)] for key, value in fields.items()]
        table = Table(table_data, colWidths=[200, 250])

        # Add table styling
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

    # Add sections to the PDF
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
        "Lateral Trunk Tilt": data.lateralTrunkTilt
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
        "Bat Shoulder Angle": data.batShoulderAng
    })

    # Add pitching breakdown in the same format...

    # Build the PDF
    doc.build(elements)

    # Move buffer position to the start
    buffer.seek(0)

    # Return the PDF as a response
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=athlete_report.pdf"}
    )

# add comment test
@app.get("/")
def health_check():
    return { "Server": "Healthy" }