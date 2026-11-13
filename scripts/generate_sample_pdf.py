"""
Script to generate a synthetic sample academic regulations PDF in English and Telugu for Phase 1 RAG verification.
"""

import os
import pymupdf  # fitz

def generate_sample_pdf(output_path: str = "sample_data/academic_regulations.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = pymupdf.open()

    font_path = "/System/Library/Fonts/KohinoorTelugu.ttc"
    font_name = "KohinoorTelugu"

    # Page 1: Attendance & Condonation Rules
    page1 = doc.new_page()
    if os.path.exists(font_path):
        page1.insert_font(fontname=font_name, fontfile=font_path)
        active_font = font_name
    else:
        active_font = "helv"

    rect1 = pymupdf.Rect(50, 50, 550, 750)
    text1 = """UNIVERSITY ACADEMIC REGULATIONS & POLICIES (2026-2027)

SECTION 1: ATTENDANCE REQUIREMENTS AND CONDONATION POLICY

1.1 Minimum Attendance Requirement:
Every registered student must maintain a minimum attendance of 75% in aggregate across all registered theory and laboratory courses in a semester to be eligible to appear for the End Semester Examinations.

1.2 Condonation of Attendance Shortage:
The Academic Council may grant condonation of attendance shortage between 65% and 75% on genuine medical grounds, supported by a valid medical certificate from an authorized medical practitioner submitted within 3 working days of resumption.

1.3 Detention Rule:
Students with less than 65% aggregate attendance shall be detained and must repeat the semester in the subsequent academic year. No condonation shall be granted under any circumstances below 65%.

తెలుగు వివరణ (Telugu Summary):
విశ్వవిద్యాలయ నిబంధనల ప్రకారం సెమిస్టర్ పరీక్షలకు హాజరు కావడానికి ప్రతి విద్యార్థికి కనీసం 75% హాజరు (75% Minimum Attendance) తప్పనిసరి. అనారోగ్య కారణాల వల్ల 65% నుండి 75% వరకు హాజరు ఉన్నవారికి మెడికల్ సర్టిఫికేట్ సమర్పిస్తే మినహాయింపు (condonation) లభిస్తుంది. 65% కంటే తక్కువ హాజరు ఉన్న విద్యార్థులు డిటైన్ చేయబడతారు.
"""
    page1.insert_textbox(rect1, text1, fontsize=11, fontname=active_font)

    # Page 2: Grading System & Credit Framework
    page2 = doc.new_page()
    if os.path.exists(font_path):
        page2.insert_font(fontname=font_name, fontfile=font_path)
    rect2 = pymupdf.Rect(50, 50, 550, 750)
    text2 = """UNIVERSITY ACADEMIC REGULATIONS & POLICIES (2026-2027)

SECTION 2: GRADING SCALE AND EVALUATION CRITERIA

2.1 Ten-Point Grading Scale:
Performance in every course is assessed using a 10-point Letter Grading Scale:
- Grade O (Outstanding): 90% to 100% (Grade Point: 10)
- Grade A+ (Excellent): 80% to 89% (Grade Point: 9)
- Grade A (Very Good): 70% to 79% (Grade Point: 8)
- Grade B+ (Good): 60% to 69% (Grade Point: 7)
- Grade B (Above Average): 50% to 59% (Grade Point: 6)
- Grade C (Pass): 40% to 49% (Grade Point: 5)
- Grade F (Fail): Below 40% (Grade Point: 0)

2.2 Passing Standard:
A student must secure a minimum of 40% marks in the End Semester Examination and 40% marks in the internal assessment independently to clear the course.

2.3 Cumulative Grade Point Average (CGPA):
CGPA is calculated by dividing the sum of (Course Credit x Grade Point) by the total number of credits registered. A minimum CGPA of 5.0 is required for degree award.
"""
    page2.insert_textbox(rect2, text2, fontsize=11, fontname=active_font)

    # Page 3: Examination Registration, Fees & Revaluation
    page3 = doc.new_page()
    if os.path.exists(font_path):
        page3.insert_font(fontname=font_name, fontfile=font_path)
    rect3 = pymupdf.Rect(50, 50, 550, 750)
    text3 = """UNIVERSITY ACADEMIC REGULATIONS & POLICIES (2026-2027)

SECTION 3: EXAMINATION REGISTRATION, FEES AND REVALUATION

3.1 Examination Fee Deadlines:
- Standard registration without late fee: October 15, 2026.
- Late registration with a fine of Rs. 500: October 22, 2026.
- Tatkal registration with a fine of Rs. 2000: Up to 3 days before exam commencement.

3.2 Revaluation & Script Recounting:
Students seeking revaluation of theory answer scripts must apply through the university portal within 10 calendar days of result declaration.
- Revaluation Fee: Rs. 1000 per theory subject.
- Recounting Fee: Rs. 300 per subject.
- Challenge Evaluation Fee: Rs. 5000 (Refundable if the score increases by >= 15%).

3.3 Malpractice Penalties:
Possession of unauthorized electronic devices or chits in the examination hall leads to cancellation of the specific examination and debarment from remaining subjects for that semester.
"""
    page3.insert_textbox(rect3, text3, fontsize=11, fontname=active_font)

    doc.save(output_path)
    doc.close()
    print(f"Generated sample PDF at: {output_path}")

if __name__ == "__main__":
    generate_sample_pdf()
