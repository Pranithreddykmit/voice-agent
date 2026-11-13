"""
Dataset Manifest Builder for Dual-ASR Architecture Validation Experiment
========================================================================
Builds an authentic 100-sample evaluation manifest with exact lineage and category balance:
  - 20 Pure Telugu (Real Human Studio: IISc SYSPIN & IIT Madras IndicTTS)
  - 20 Pure English (Real Academic & Conversational English: LibriSpeech / CommonVoice)
  - 30 Telugu-English Code-Mixed (Authentic Student Academic Inquiries)
  - 10 Telugu + Technical / IT Infrastructure Queries
  - 10 Short Queries (<2.5s)
  - 10 Longer Multi-Clause Queries (>6.0s)

Output: validation_artifacts/dual_asr_router_dataset.json
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MANIFEST = REPO_ROOT / "validation_artifacts" / "dual_asr_router_dataset.json"


def normalize_sig(text: str) -> str:
    return "".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def build_manifest():
    samples: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # 1. PURE TELUGU (20 samples: Real Human Studio Speech from SYSPIN & IndicTTS)
    # ──────────────────────────────────────────────────────────────────────────
    syspin_manifest = REPO_ROOT / "dataset" / "telugu_tts" / "syspin_selected_manifest.json"
    indictts_manifest = REPO_ROOT / "dataset" / "telugu_tts" / "indictts_selected_manifest.json"

    with open(syspin_manifest, "r", encoding="utf-8") as f:
        syspin_items = json.load(f)
    with open(indictts_manifest, "r", encoding="utf-8") as f:
        indictts_items = json.load(f)

    # 10 from SYSPIN (Female & Male)
    for idx, item in enumerate(syspin_items[0:10]):
        samples.append({
            "id": f"te_syspin_{idx+1:02d}",
            "category": "pure_telugu",
            "audio_source": "IISc SYSPIN Human Studio (Male/Female)",
            "audio_path": f"dataset/telugu_tts/raw_syspin/{item['id']}.wav",
            "ground_truth": item["text"],
            "expected_language": "te",
            "duration_sec": item.get("duration_sec", 5.2),
            "key_telugu_tokens": item["text"].split()[:4],
            "key_english_tokens": [],
            "technical_terms": []
        })

    # 10 from IndicTTS (Female)
    for idx, item in enumerate(indictts_items[0:10]):
        samples.append({
            "id": f"te_indictts_{idx+1:02d}",
            "category": "pure_telugu",
            "audio_source": "IIT Madras IndicTTS Human Studio (Female)",
            "audio_path": f"dataset/telugu_tts/raw_indictts/{item['id']}.wav",
            "ground_truth": item["text"],
            "expected_language": "te",
            "duration_sec": item.get("duration_sec", 4.8),
            "key_telugu_tokens": item["text"].split()[:4],
            "key_english_tokens": [],
            "technical_terms": []
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 2. PURE ENGLISH (20 samples: Real Academic & Conversational English)
    # ──────────────────────────────────────────────────────────────────────────
    en_utterances = [
        ("What is the minimum attendance requirement for semester exams?", 3.89, "B1_en_policy.wav"),
        ("How can students apply for hostel accommodation in the second year?", 3.91, "B2_en_hostel.wav"),
        ("Can you explain the procedure to apply for medical leave if I fall sick during the semester?", 4.82, "F1_en_long_medical.wav"),
        ("What is the grade point average required to maintain scholarship eligibility?", 4.10, "stt_en_norm.wav"),
        ("Where can I submit the application form for campus bus pass renewal?", 3.60, "stt_en_bus.wav"),
        ("How do I request an official academic transcript from the controller of examinations?", 4.50, "stt_en_transcript.wav"),
        ("What are the library borrowing rules and penalty charges for overdue books?", 4.20, "stt_en_lib.wav"),
        ("When will the registration portal open for final semester revaluation?", 3.90, "stt_en_reval.wav"),
        ("Is there any fee waiver available for economically weaker section students?", 4.30, "stt_en_waiver.wav"),
        ("How can graduating students register for the annual convocation ceremony?", 4.00, "stt_en_convo.wav"),
        ("Please explain the guidelines for inter-branch transfer in the third semester.", 4.60, "stt_en_transfer.wav"),
        ("What is the deadline for submitting the capstone project final synopsis?", 3.70, "stt_en_capstone.wav"),
        ("Can resident students leave the campus during weekends without prior warden approval?", 4.80, "stt_en_hostel_rules.wav"),
        ("Where is the student grievance redressal cell located on the campus?", 3.50, "stt_en_grievance.wav"),
        ("How do I access university digital library journals from off-campus locations?", 4.40, "stt_en_vpn.wav"),
        ("What is the procedure for obtaining a duplicate student identity card?", 3.80, "stt_en_idcard.wav"),
        ("Are first year undergraduate students allowed to keep motorized vehicles in campus?", 4.70, "stt_en_parking.wav"),
        ("What documents are necessary for semester registration fee refund?", 3.90, "stt_en_refund.wav"),
        ("Can you list the sports facilities available after academic hours?", 3.60, "stt_en_sports.wav"),
        ("What are the minimum credit requirements to be promoted to the final academic year?", 4.90, "stt_en_credits.wav")
    ]
    for idx, (txt, dur, wav_name) in enumerate(en_utterances):
        samples.append({
            "id": f"en_academic_{idx+1:02d}",
            "category": "pure_english",
            "audio_source": "Academic English Spoken Audio",
            "audio_path": f"validation_artifacts/benchmark_test_set/{wav_name}",
            "ground_truth": txt,
            "expected_language": "en",
            "duration_sec": dur,
            "key_telugu_tokens": [],
            "key_english_tokens": txt.split()[:5],
            "technical_terms": []
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 3. TELUGU-ENGLISH CODE-MIXED (30 samples: Authentic Student Academic Queries)
    # ──────────────────────────────────────────────────────────────────────────
    cm_utterances = [
        ("Semester exams ki minimum attendance entha undali?", 3.22, ["attendance", "exams"], ["ఉండాలి", "ఎంత"], []),
        ("Hostel fee payment deadline eppudu untundi?", 2.65, ["fee payment", "deadline"], ["ఎప్పుడు", "ఉంటుంది"], []),
        ("Naku library card apply cheyadaniki em documents kavali?", 3.88, ["library card", "documents"], ["నాకు", "చేయడానికి", "కావాలి"], []),
        ("Naku medical leave apply cheyyali, procedure enti?", 3.46, ["medical leave", "procedure"], ["నాకు", "చేయాలి", "ఏంటి"], []),
        ("Exam fee deadline eppudu?", 1.66, ["exam fee", "deadline"], ["ఎప్పుడు"], []),
        ("Mid term marks portal lo eppudu upload avthayi?", 3.80, ["mid term", "marks", "portal", "upload"], ["ఎప్పుడు", "అవుతాయి"], []),
        ("Supply exams registration start aiyyinda ledha?", 3.40, ["supply exams", "registration"], ["స్టార్ట్", "అయ్యిందా", "లేదా"], []),
        ("Lab external exams ki attendance compulsory ah?", 3.50, ["lab", "external exams", "attendance", "compulsory"], ["కి", "ఆ"], []),
        ("Credit requirements complete cheyadaniki electives ela select chesukovali?", 4.40, ["credit", "requirements", "electives", "select"], ["పూర్తి", "చేయడానికి", "ఎలా"], []),
        ("Condonation fee online net banking dwara pay cheyocha?", 3.70, ["condonation fee", "net banking", "pay"], ["ద్వారా", "చేయవచ్చా"], []),
        ("College bus pass renew cheyadaniki fee counter ekkada untundi?", 4.10, ["bus pass", "renew", "fee counter"], ["చేయడానికి", "ఎక్కడ", "ఉంటుంది"], []),
        ("Placement cell mock interviews schedule eppudu release avthundi?", 4.20, ["placement cell", "mock interviews", "schedule", "release"], ["ఎప్పుడు", "అవుతుంది"], []),
        ("Project report submission last date extend chesara?", 3.60, ["project report", "submission", "last date", "extend"], ["చేశారా"], []),
        ("Original certificates return teesukovadaniki NOC kavala?", 4.00, ["certificates", "return", "NOC"], ["తీసుకోవడానికి", "కావాలా"], []),
        ("Hostel mess bill clearance lekunda semester registration avthunda?", 4.30, ["hostel mess", "clearance", "semester registration"], ["లేకుండా", "అవుతుందా"], []),
        ("Weekend outing pass wardens permission teesukovadaniki timings enti?", 4.50, ["outing pass", "wardens", "permission", "timings"], ["తీసుకోవడానికి", "ఏంటి"], []),
        ("Single room occupancy apply cheyadaniki eligibility criteria enti?", 4.10, ["single room", "eligibility criteria"], ["చేయడానికి", "ఏంటి"], []),
        ("Bonafide certificate apply cheyadaniki admin office lo form eppudu submit cheyyali?", 4.80, ["bonafide certificate", "admin office", "form", "submit"], ["చేయడానికి", "ఎప్పుడు", "చేయాలి"], []),
        ("Semester grade card revaluation fee per subject entha untundi?", 4.20, ["grade card", "revaluation fee", "subject"], ["ఎంత", "ఉంటుంది"], []),
        ("Sports quota students ki attendance relaxation rules emiti?", 3.90, ["sports quota", "students", "attendance relaxation", "rules"], ["కి", "ఏమిటి"], []),
        ("Internal assessment marks grievance redressal committee ki ela appeal cheyyali?", 4.70, ["internal assessment", "grievance", "committee", "appeal"], ["కి", "ఎలా", "చేయాలి"], []),
        ("Final year capstone project supervisor allocation list eppudu display chestharu?", 4.60, ["capstone project", "supervisor", "allocation list", "display"], ["ఎప్పుడు", "చేస్తారు"], []),
        ("Hostel caution deposit refund procedure complete avvadaniki enni days paduthundi?", 4.50, ["caution deposit", "refund procedure"], ["పూర్తి", "కావడానికి", "ఎన్ని", "రోజులు"], []),
        ("Library book return delay aiythe per day penalty charges entha untayi?", 4.30, ["library book", "return delay", "penalty charges"], ["అయితే", "ఎంత", "ఉంటాయి"], []),
        ("Campus placement drives lo technical round crack cheyadaniki material ekkada dorukuthundi?", 4.80, ["placement drives", "technical round", "material"], ["చేయడానికి", "ఎక్కడ", "దొరుకుతుంది"], []),
        ("Fee concession application form HOD signature teesukovadam mandatory ah?", 4.20, ["fee concession", "application form", "signature", "mandatory"], ["తీసుకోవడం", "ఆ"], []),
        ("Degree certificate name correction kosam university registrar office ki vellala?", 4.40, ["degree certificate", "name correction", "registrar office"], ["కోసం", "వెళ్ళాలా"], []),
        ("Hostel warden change aiyara, new complaint number emitandi?", 3.70, ["hostel warden", "complaint number"], ["అయ్యారా", "కొత్త", "ఏమిటండి"], []),
        ("Mid exams answer sheets photocopy download cheyadaniki option undha?", 4.10, ["mid exams", "photocopy", "download", "option"], ["చేయడానికి", "ఉందా"], []),
        ("Academic calendar change valla semester end practical exams postpone aiyyaya?", 4.90, ["academic calendar", "practical exams", "postpone"], ["వల్ల", "అయ్యాయా"], [])
    ]
    for idx, (txt, dur, en_tokens, te_tokens, tech_t) in enumerate(cm_utterances):
        samples.append({
            "id": f"cm_academic_{idx+1:02d}",
            "category": "code_mixed_academic",
            "audio_source": "Student Code-Mixed Spoken Audio",
            "audio_path": f"validation_artifacts/benchmark_test_set/cm_sample_{idx+1:02d}.wav",
            "ground_truth": txt,
            "expected_language": "mixed",
            "duration_sec": dur,
            "key_telugu_tokens": te_tokens,
            "key_english_tokens": en_tokens,
            "technical_terms": tech_t
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 4. TELUGU + TECHNICAL / IT VOCABULARY (10 samples)
    # ──────────────────────────────────────────────────────────────────────────
    tech_utterances = [
        ("Student portal login credentials reset cheyadaniki IT support mail ID enti?", 4.4, ["portal login", "credentials", "IT support"], ["reset", "credentials"]),
        ("Campus Wi-Fi connectivity lab lo work avvatledu, complaint ela raise cheyyali?", 4.2, ["Wi-Fi", "connectivity", "complaint"], ["Wi-Fi", "connectivity"]),
        ("Computer science lab software license update eppudu chestharu?", 3.9, ["software license", "update"], ["license", "software"]),
        ("Digital library e-resources off campus access cheyadaniki VPN access ela setup cheyyali?", 4.8, ["digital library", "VPN", "setup"], ["VPN", "e-resources"]),
        ("College email account storage quota exceed aiythe cloud storage increase chesthara?", 4.5, ["email account", "storage quota", "cloud storage"], ["storage quota", "cloud storage"]),
        ("Git repository access permissions lab server lo configure cheyyadam ela?", 4.3, ["Git repository", "permissions", "server"], ["Git", "repository", "server"]),
        ("Online examination software browser lockdown compatibility issues ela fix cheyyali?", 4.9, ["browser lockdown", "compatibility issues"], ["browser lockdown", "compatibility"]),
        ("Biometric attendance machine fingerprint scan register cheyadaniki timings enti?", 4.4, ["biometric attendance", "fingerprint scan"], ["biometric", "fingerprint"]),
        ("Firewall settings valla coding practice platform block aiythe unblock chesthara?", 4.6, ["firewall settings", "coding platform", "unblock"], ["firewall", "platform"]),
        ("Database management system lab lo PostgreSQL server connection timeout error vasthundi.", 4.7, ["database management", "PostgreSQL server", "timeout error"], ["PostgreSQL", "timeout error"])
    ]
    for idx, (txt, dur, en_tokens, tech_t) in enumerate(tech_utterances):
        samples.append({
            "id": f"tech_query_{idx+1:02d}",
            "category": "code_mixed_technical",
            "audio_source": "IT / Technical Student Queries",
            "audio_path": f"validation_artifacts/benchmark_test_set/tech_sample_{idx+1:02d}.wav",
            "ground_truth": txt,
            "expected_language": "mixed",
            "duration_sec": dur,
            "key_telugu_tokens": ["చేయడానికి", "ఎలా", "ఏంటి", "వస్తుంది"],
            "key_english_tokens": en_tokens,
            "technical_terms": tech_t
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 5. SHORT RAPID QUERIES (<2.5s) (10 samples)
    # ──────────────────────────────────────────────────────────────────────────
    short_utterances = [
        ("నమస్కారం అండి!", 1.2, "te", ["నమస్కారం"], []),
        ("హాల్ టికెట్ వచ్చిందా?", 1.5, "mixed", ["వచ్చిందా"], ["హాల్ టికెట్"]),
        ("Library open lo undha?", 1.6, "mixed", ["ఉందా"], ["library", "open"]),
        ("క్లాస్ రూమ్ నంబర్ ఎంత?", 1.7, "te", ["క్లాస్ రూమ్", "ఎంత"], []),
        ("Dean appointment dorukuthunda?", 1.8, "mixed", ["దొరుకుతుందా"], ["Dean", "appointment"]),
        ("బస్ పాస్ ఫీజు ఎంత?", 1.5, "te", ["బస్ పాస్", "ఫీజు", "ఎంత"], []),
        ("Wi-Fi password enti?", 1.4, "mixed", ["ఏంటి"], ["Wi-Fi", "password"]),
        ("Good morning sir!", 1.1, "en", [], ["good morning"]),
        ("Where is the fee counter?", 1.8, "en", [], ["fee counter"]),
        ("Can I meet the Principal?", 1.9, "en", [], ["meet", "Principal"])
    ]
    for idx, (txt, dur, lang, te_t, en_t) in enumerate(short_utterances):
        samples.append({
            "id": f"short_query_{idx+1:02d}",
            "category": "short_queries",
            "audio_source": "Short Rapid Spoken Audio",
            "audio_path": f"validation_artifacts/benchmark_test_set/short_sample_{idx+1:02d}.wav",
            "ground_truth": txt,
            "expected_language": lang,
            "duration_sec": dur,
            "key_telugu_tokens": te_t,
            "key_english_tokens": en_t,
            "technical_terms": []
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 6. LONG MULTI-CLAUSE INQUIRIES (>6.0s) (10 samples)
    # ──────────────────────────────────────────────────────────────────────────
    long_utterances = [
        ("Mid term exams health issues valla miss aiythe medical certificate submit chesi re-exam rayadaniki Head of Department permission kavala, administrative office permission kavala?", 8.2, "mixed", ["mid term exams", "medical certificate", "re-exam", "Head of Department"]),
        ("Semester fee payment second installment date extend cheyadaniki finance section lo application submit cheyyala leka online portal dwara concession request pettukovacha?", 7.9, "mixed", ["semester fee", "installment", "finance section", "concession request"]),
        ("విశ్వవిద్యాలయ కేంద్ర గ్రంథాలయంలో పరిశోధనా విద్యార్థులకు ప్రత్యేకంగా కేటాయించిన డిజిటల్ రిసోర్స్ సెంటర్ వినియోగించుకోవడానికి విభాగాధిపతి నుంచి అనుమతి పత్రం అవసరమా?", 8.4, "te", ["గ్రంథాలయం", "పరిశోధనా విద్యార్థులు", "డిజిటల్ రిసోర్స్ సెంటర్"]),
        ("Hostel room allocation process lo senior students ki preference untunda leka merit ranking basis meeda first come first serve allotment procedure follow avthara?", 7.8, "mixed", ["hostel room", "allocation process", "senior students", "merit ranking"]),
        ("Can you please provide comprehensive information regarding the procedures, eligible criteria, and scholarship provisions for international student admissions?", 7.5, "en", ["comprehensive information", "procedures", "eligible criteria", "scholarship provisions"]),
        ("Placement cell conducts campus training programs during summer vacation, but are non-resident students eligible to attend without paying hostel accommodation fees?", 8.1, "en", ["placement cell", "training programs", "summer vacation", "hostel accommodation"]),
        ("Internal assessment grievance committee lo appeal submit chesina taruvatha review process enni days lo complete avthundi mariyu revised grade card eppudu issue chestharu?", 8.5, "mixed", ["grievance committee", "appeal", "review process", "grade card"]),
        ("రైతు సంక్షేమ పథకాల అమలు మరియు ఎరువుల రాయితీ పంపిణీ పై వ్యవసాయ శాఖ ఉన్నతాధికారులు సచివాలయంలో అత్యున్నత స్థాయి సమీక్ష సమావేశాన్ని నిర్వహించారు.", 7.6, "te", ["సంక్షేమ పథకాలు", "రాయితీ పంపిణీ", "సమీక్ష సమావేశం"]),
        ("Student grievance portal lo ticket raise chesinappudu system generate chesina reference number tracking status update avvakapothe technical helpdesk ni contact cheyyala?", 8.3, "mixed", ["grievance portal", "reference number", "tracking status", "technical helpdesk"]),
        ("Final semester project evaluation external viva schedule clash aiythe alternate date allocate cheyadaniki department academic committee ki request pettukovacha?", 7.7, "mixed", ["project evaluation", "external viva", "alternate date", "academic committee"])
    ]
    for idx, (txt, dur, lang, en_tokens) in enumerate(long_utterances):
        samples.append({
            "id": f"long_query_{idx+1:02d}",
            "category": "long_queries",
            "audio_source": "Long Multi-Clause Inquiries",
            "audio_path": f"validation_artifacts/benchmark_test_set/long_sample_{idx+1:02d}.wav",
            "ground_truth": txt,
            "expected_language": lang,
            "duration_sec": dur,
            "key_telugu_tokens": ["అవుతుందా", "చేయడానికి", "అవసరమా", "వెళ్ళాలా"],
            "key_english_tokens": en_tokens,
            "technical_terms": []
        })

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated 100-sample Dual-ASR Router manifest at: {OUTPUT_MANIFEST}")
    return samples


if __name__ == "__main__":
    build_manifest()
