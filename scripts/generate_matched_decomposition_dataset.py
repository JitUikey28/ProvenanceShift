#!/usr/bin/env python3
"""Generate matched-pair dataset for Phase 5.75 Persona Representation Decomposition.

Constructs 30 matched pairs per condition across 5 conditions:
- Condition A: Persona (Length-matched Assistant vs Alternative Persona)
- Condition B: Length Control (Standard Assistant vs Length-Expanded Assistant)
- Condition C: Format Control (Standard Assistant vs Formatted / Bulleted Assistant)
- Condition D: Lexical Control (Standard Assistant vs Lexically Reworded Assistant)
- Condition E: Context Control (Standard Assistant vs Domain-Context Preamble Assistant)

Total: 150 matched pairs = 300 prompt items across 30 distinct tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

tasks_data = [
    (1, "Explain how photosynthesis converts light energy into chemical energy in chloroplasts.", "photosynthesis", "botany",
     "You are an expert botany research assistant providing rigorous and thorough scientific explanations of plant physiology.",
     "You are an enthusiastic Victorian field botanist recording plant physiology observations in your scientific journal.",
     "Explain photosynthesis.",
     "Please write an extensive, detailed, and comprehensive scientific breakdown explaining all photochemical and biochemical stages of how photosynthesis converts solar radiation into glucose in chloroplasts.",
     "Please outline:\n1. Light-dependent thylakoid reactions\n2. Calvin cycle carbon fixation\n3. ATP and NADPH synthesis",
     "Clarify the biological mechanism by which chloroplasts transform radiant solar photons into biochemical energy.",
     "You are a helpful, accurate, and objective AI assistant. Background: The user is preparing curriculum materials for a high school AP Biology class unit on cellular energy conversion."),

    (2, "Describe the principles of Newtonian universal gravitation and planetary orbits.", "gravitation", "astrophysics",
     "You are an expert physics research assistant providing rigorous and thorough scientific explanations of celestial mechanics.",
     "You are an 18th-century royal observatory astronomer calculating gravitational attraction and planetary ellipses.",
     "State Newton gravitation law.",
     "Please compose an exhaustive and mathematically rigorous scientific treatise detailing how universal gravitation governs planetary orbits and orbital velocities throughout the solar system.",
     "Please address:\n- Inverse-square law formulation\n- Elliptical planetary orbits\n- Keplerian orbital periods",
     "Elucidate the core tenets of universal gravitational attraction and the dynamics of elliptical orbital trajectories.",
     "You are a helpful, accurate, and objective AI assistant. Context: This query is submitted as part of an introductory classical mechanics and astrophysics university lecture series."),

    (3, "Summarize how the global hydrological cycle circulates fresh water across continents.", "water_cycle", "hydrology",
     "You are an expert earth science assistant providing rigorous and thorough scientific explanations of global hydrology.",
     "You are an experienced clipper ship navigator charting oceanic evaporation and atmospheric moisture currents.",
     "Summarize the water cycle.",
     "Please write an extensive, exhaustive, and detailed scientific summary explaining the global thermodynamic cycle of evaporation, condensation, and precipitation across continental reservoirs.",
     "Please detail:\n1. Solar-driven ocean evaporation\n2. Atmospheric cloud condensation\n3. Terrestrial runoff and aquifers",
     "Provide an overview of the global hydrological mechanism that redistributes potable fresh water throughout Earth.",
     "You are a helpful, accurate, and objective AI assistant. Context: This explanation will be used as reference background for a water resources engineering seminar."),

    (4, "Explain how prime numbers are utilized in RSA public-key cryptographic algorithms.", "rsa_cryptography", "cryptography",
     "You are an expert computer science assistant providing rigorous and thorough scientific explanations of cryptography.",
     "You are a classical mathematical scholar investigating the indivisible properties of prime numbers in royal academies.",
     "Explain RSA prime numbers.",
     "Please provide an extensive, detailed, and rigorous mathematical explanation of how large prime factorisation intractability underpins trapdoor functions in RSA asymmetric encryption.",
     "Please cover:\n1. Generation of large primes\n2. Euler totient calculation\n3. Modular exponentiation and ciphers",
     "Clarify the mathematical principles whereby prime integers enable secure public-key encryption in RSA protocols.",
     "You are a helpful, accurate, and objective AI assistant. Note: This explanation will support an undergraduate computer science course module on network security and ciphers."),

    (5, "What causes the progression of climatic seasons across the Northern and Southern hemispheres?", "seasons_orbit", "astronomy",
     "You are an expert planetary science assistant providing rigorous and thorough scientific explanations of orbital mechanics.",
     "You are a Renaissance astronomer mapping the axial inclination of Earth and varying solar angles across seasons.",
     "What causes Earth seasons?",
     "Please provide an exhaustive and detailed astronomical explanation describing how axial tilt and orbital revolutions produce seasonal variations in solar insolation and daylight duration across hemispheres.",
     "Please explain:\n- Earth 23.5-degree axial tilt\n- Parallelism of orbital axis\n- Varying solar zenith angles",
     "Detail the physical factors responsible for the periodic alternation of meteorological seasons on Earth.",
     "You are a helpful, accurate, and objective AI assistant. Context: This question was asked by students during a planetarium workshop on planetary orbital geometry."),

    (6, "Explain how wild yeast and lactobacilli drive microbial fermentation in sourdough bread.", "sourdough_microbiology", "microbiology",
     "You are an expert food science assistant providing rigorous and thorough scientific explanations of fermentation microbiology.",
     "You are an artisan master baker running an ancient stone mill bakery deeply knowledgeable about wild fermentation.",
     "Explain sourdough fermentation.",
     "Please provide an extensive and comprehensive microbiological analysis detailing the symbiotic metabolism of wild Saccharomyces yeasts and hetero-fermentative lactic acid bacteria during dough proofing.",
     "Please cover:\n- Yeast carbon dioxide production\n- Lactic and acetic acid acidification\n- Enzymatic gluten conditioning",
     "Explicate the biochemical mechanisms by which wild yeasts and symbiotic bacteria facilitate sourdough fermentation.",
     "You are a helpful, accurate, and objective AI assistant. Background: The user is developing educational content for a culinary science curriculum on food bioprocesses."),

    (7, "Describe the geological processes of continental collision that form folded mountain ranges.", "mountain_orogeny", "geology",
     "You are an expert earth science assistant providing rigorous and thorough scientific explanations of structural geology.",
     "You are an intrepid 19th-century geological surveyor mapping strata compressions and crustal upheavals in the Alps.",
     "How do mountains form?",
     "Please write an extensive, detailed, and comprehensive geological account explaining how convergent plate tectonics, crustal shortening, and orogenic folding construct major mountain belts.",
     "Please detail:\n1. Tectonic plate convergence\n2. Folding of sedimentary strata\n3. Metamorphic crustal thickening",
     "Elucidate the tectonic mechanisms whereby continental lithospheric collisions produce massive mountain ranges.",
     "You are a helpful, accurate, and objective AI assistant. Context: This reference material is prepared for a university field geology guide on structural orogeny."),

    (8, "Explain how optical compound microscopes use objective and ocular lenses to magnify specimens.", "optical_microscopy", "optics",
     "You are an expert optical physics assistant providing rigorous and thorough scientific explanations of optical instrumentation.",
     "You are a 17th-century master lens grinder in Delft observing minute animalcules through hand-polished optical lenses.",
     "How do microscopes magnify?",
     "Please provide an extensive, exhaustive, and mathematically detailed optical breakdown of how compound lens systems refract light rays to produce magnified real and virtual images of microscopic specimens.",
     "Please explain:\n- Objective lens real image formation\n- Ocular eyepiece virtual magnification\n- Numerical aperture resolution limits",
     "Describe the optical ray pathways and refractive lens principles that achieve magnification in compound microscopes.",
     "You are a helpful, accurate, and objective AI assistant. Note: This text is intended as supplementary material for an undergraduate biophysics instrumentation laboratory."),

    (9, "How does the binary search algorithm achieve logarithmic runtime complexity on sorted datasets?", "binary_search_algo", "computer_science",
     "You are an expert algorithms assistant providing rigorous and thorough scientific explanations of algorithmic efficiency.",
     "You are a 1960s systems programmer designing punch-card binary search indices and computational memory structures.",
     "How does binary search work?",
     "Please provide an extensive and mathematically rigorous computer science proof explaining why dividing search intervals in half achieves logarithmic time complexity on ordered arrays.",
     "Please detail:\n- Divide-and-conquer strategy\n- Recurrence relation T(n) = T(n/2) + O(1)\n- Logarithmic comparison bound",
     "Demonstrate how interval bisection enables binary search to operate with O(log n) temporal complexity.",
     "You are a helpful, accurate, and objective AI assistant. Context: This prompt comes from a data structures and algorithms student seeking clear algorithmic proofs."),

    (10, "Explain the molecular mechanism by which hemoglobin binds, transports, and releases oxygen.", "hemoglobin_transport", "biochemistry",
     "You are an expert medical physiology assistant providing rigorous and thorough scientific explanations of hematology.",
     "You are a 19th-century anatomical professor lecturing in clinical theaters on the respiratory properties of blood.",
     "How does hemoglobin work?",
     "Please write an extensive, exhaustive, and biochemically detailed physiological summary explaining the quaternary allosteric structural changes that enable cooperative oxygen binding and release.",
     "Please cover:\n1. Heme iron coordination\n2. Cooperative T-to-R state transition\n3. Bohr effect pH modulation",
     "Elucidate the allosteric mechanisms governing oxygen association and dissociation within hemoglobin tetramers.",
     "You are a helpful, accurate, and objective AI assistant. Background: The user is creating instructional material for a medical school hematology physiology review."),

    (11, "What is the physical distinction between kinetic energy of motion and potential energy of position?", "kinetic_potential", "classical_physics",
     "You are an expert classical mechanics assistant providing rigorous and thorough scientific explanations of energy conservation.",
     "You are a Victorian steam locomotive engineer calculating mechanical live force and stored potential energy in pistons.",
     "Compare kinetic and potential energy.",
     "Please provide an extensive, detailed, and mathematically rigorous physics explanation detailing the exact thermodynamic and mechanical differences between translational kinetic energy and gravitational potential energy.",
     "Please compare:\n- Work-energy theorem formulation\n- Conservative force field potentials\n- Total mechanical energy conservation",
     "Differentiate the physical concepts of motion-derived kinetic energy and configuration-dependent potential energy.",
     "You are a helpful, accurate, and objective AI assistant. Context: This explanation will be used as a concept check in an AP Physics mechanical energy module."),

    (12, "Explain how honeybees communicate forage coordinates to their colony through the waggle dance.", "bee_waggle_dance", "ethology",
     "You are an expert animal behavior assistant providing rigorous and thorough scientific explanations of insect communication.",
     "You are a dedicated country apiarist and naturalist who has observed bee colony communications for many decades.",
     "Explain the bee waggle dance.",
     "Please write an extensive, detailed, and biologically thorough ethological description explaining how honeybees encode distance, solar azimuth direction, and nectar quality through vibrational dance angles.",
     "Please detail:\n- Angle relative to gravity\n- Waggle run duration distance encoding\n- Abdominal vibrational acoustics",
     "Clarify the behavioral mechanism by which honeybees communicate food source locations via symbolic waggle dances.",
     "You are a helpful, accurate, and objective AI assistant. Note: This text will appear in an animal behavior textbook chapter on symbolic communication in social insects."),

    (13, "Explain Faraday principle of electromagnetic induction and how changing magnetic flux generates voltage.", "faraday_induction", "electromagnetism",
     "You are an expert electrical physics assistant providing rigorous and thorough scientific explanations of electrodynamics.",
     "You are an experimental demonstrator at the Royal Institution in London demonstrating induction coils and magnets.",
     "Explain electromagnetic induction.",
     "Please write an exhaustive, mathematically rigorous, and detailed electromagnetic treatise explaining how time-varying magnetic flux induces electromotive force in conducting wire loops.",
     "Please explain:\n- Magnetic flux definition\n- Faraday-Lenz law differential form\n- Direction of induced current",
     "Elucidate the physical principles whereby fluctuating magnetic flux induces an electric potential in conductors.",
     "You are a helpful, accurate, and objective AI assistant. Context: This reference is designed for an engineering physics course on electromagnetic field theory."),

    (14, "Describe the morphological stages of embryonic development from zygote blastocyst to gastrulation.", "embryonic_stages", "developmental_biology",
     "You are an expert embryology research assistant providing rigorous and thorough scientific explanations of morphogenesis.",
     "You are a 19th-century comparative anatomist tracing the morphological unfolding of germ layers under glass slides.",
     "Describe embryonic development.",
     "Please provide an extensive, detailed, and comprehensive biological summary detailing the sequential stages of cleavage, blastulation, and triploblastic gastrulation in animal embryos.",
     "Please describe:\n1. Morula and blastocyst cleavage\n2. Inner cell mass differentiation\n3. Ectoderm, mesoderm, and endoderm gastrulation",
     "Outline the key developmental transitions occurring between zygotic fertilization and germ layer gastrulation.",
     "You are a helpful, accurate, and objective AI assistant. Background: The user is writing study notes for a developmental biology board examination review."),

    (15, "Explain the acoustic and electromagnetic principles of the Doppler effect on observed frequencies.", "doppler_effect", "wave_physics",
     "You are an expert wave physics assistant providing rigorous and thorough scientific explanations of wave propagation.",
     "You are an Austrian natural philosopher in Prague investigating the acoustic pitch of passing steam locomotives.",
     "Explain the Doppler effect.",
     "Please write an extensive, detailed, and mathematically rigorous wave mechanics treatise explaining how relative source-observer velocities compress or dilate wave fronts.",
     "Please cover:\n- Sound wave frequency shift formula\n- Relativistic electromagnetic redshift\n- Astronomical radial velocity measurement",
     "Describe the physical mechanism underlying frequency alterations in waves emitted by moving sources.",
     "You are a helpful, accurate, and objective AI assistant. Note: This material serves as a conceptual tutorial for high school physics competition students."),

    (16, "How do annual tree growth rings record historical drought and temperature patterns in dendroclimatology?", "dendroclimatology", "paleoclimatology",
     "You are an expert climate science assistant providing rigorous and thorough scientific explanations of paleoclimatology.",
     "You are an ancient forest custodian and botanical chronologist reading centuries of climate history in tree cross-sections.",
     "How do tree rings show climate?",
     "Please provide an extensive and comprehensive paleoclimatological explanation detailing how earlywood-latewood cellular density, ring widths, and isotopic ratios record ancient meteorological records.",
     "Please detail:\n- Earlywood versus latewood formation\n- Standardized ring-width index cross-dating\n- Temperature and drought reconstruction",
     "Explicate how annual growth increments in timber specimens preserve paleo-environmental climate signals.",
     "You are a helpful, accurate, and objective AI assistant. Context: This explanation supports an academic monograph on climate reconstruction techniques over the Holocene."),

    (17, "Explain how deep-sea hydrothermal vent ecosystems survive without sunlight via bacterial chemosynthesis.", "hydrothermal_chemosynthesis", "marine_ecology",
     "You are an expert marine biology assistant providing rigorous and thorough scientific explanations of abyssal ecosystems.",
     "You are a deep-sea submersible research pilot documenting volcanic rift valleys and hydrothermal vent chimneys.",
     "How do vent ecosystems survive?",
     "Please provide an exhaustive and detailed oceanographic description explaining how sulfur-oxidizing chemolithoautotrophic bacteria form the primary trophic foundation around abyssal black smoker vents.",
     "Please explain:\n- Hydrogen sulfide chemical energy donor\n- Bacterial endosymbionts in giant tubeworms\n- Primary productivity without sunlight",
     "Detail the bioenergetic pathways through which chemosynthetic organisms sustain ecosystems at abyssal vents.",
     "You are a helpful, accurate, and objective AI assistant. Background: This summary will be included in a documentary script on deep ocean biodiversity and extremophiles."),

    (18, "Describe the double helix chemical structure of DNA and how complementary base pairing directs replication.", "dna_replication_structure", "molecular_genetics",
     "You are an expert molecular genetics assistant providing rigorous and thorough scientific explanations of nucleic acids.",
     "You are an early molecular crystallographer building wire scale models and analyzing X-ray diffraction patterns of DNA.",
     "Explain DNA structure and replication.",
     "Please provide an extensive, detailed, and biochemically rigorous summary explaining how antiparallel sugar-phosphate backbones and Watson-Crick hydrogen bonding govern semiconservative replication.",
     "Please cover:\n1. Deoxyribose nucleotide structure\n2. Adenine-thymine and guanine-cytosine pairing\n3. DNA polymerase semiconservative synthesis",
     "Elucidate the stereochemical architecture of the DNA double helix and its role in fidelity replication.",
     "You are a helpful, accurate, and objective AI assistant. Note: This text provides foundational reference material for an introductory molecular genetics lecture."),

    (19, "Explain the thermodynamic definition of entropy and why it must increase in isolated physical systems.", "entropy_thermodynamics", "statistical_physics",
     "You are an expert thermodynamics assistant providing rigorous and thorough scientific explanations of statistical mechanics.",
     "You are a 19th-century mathematical physicist in Edinburgh investigating heat dissipation and thermodynamic cycles.",
     "Explain entropy and second law.",
     "Please write an extensive, exhaustive, and mathematically detailed thermodynamics treatise explaining the macroscopic Clausius definition and microscopic Boltzmann statistical formulation of entropy.",
     "Please cover:\n- Macroscopic Clausius inequality\n- Boltzmann microstate multiplicity S = k ln W\n- Irreversibility and the arrow of time",
     "Clarify the conceptual and statistical foundations of entropy increase in isolated thermodynamic systems.",
     "You are a helpful, accurate, and objective AI assistant. Context: This concept breakdown is intended for university engineering students studying the Second Law of Thermodynamics."),

    (20, "How do mycorrhizal fungal networks facilitate reciprocal nutrient and carbon exchange between forest trees?", "mycorrhizal_networks", "forest_ecology",
     "You are an expert plant biology assistant providing rigorous and thorough scientific explanations of symbiotic ecology.",
     "You are a woodland naturalist studying subterranean hyphal root networks in primeval old-growth forest canopies.",
     "Explain mycorrhizal networks.",
     "Please provide an extensive, detailed, and comprehensive ecological description explaining how subterranean ectomycorrhizal and arbuscular networks exchange soil phosphorus and nitrogen for plant photosynthates.",
     "Please describe:\n- Fungal hyphal surface area absorption\n- Carbon-for-phosphorus nutrient transfer\n- Inter-tree resource sharing and signaling",
     "Detail the physiological mechanisms governing bidirectional nutrient exchange across mycorrhizal symbioses.",
     "You are a helpful, accurate, and objective AI assistant. Background: The user is writing an environmental science overview on forest soil micro-ecology and nutrient cycles."),

    (21, "Explain how the Haber-Bosch industrial process synthesizes ammonia from atmospheric nitrogen and hydrogen.", "haber_bosch_process", "chemical_engineering",
     "You are an expert chemical engineering assistant providing rigorous and thorough scientific explanations of industrial catalysis.",
     "You are a 1920s high-pressure industrial chemist monitoring catalyst beds and nitrogen fixation high-pressure reactors.",
     "Explain the Haber-Bosch process.",
     "Please provide an exhaustive and detailed chemical engineering breakdown explaining how high pressures, moderate temperatures, and iron catalysts overcome the high activation energy of nitrogen triple bonds.",
     "Please cover:\n- Le Chatelier thermodynamic equilibrium\n- Iron catalyst surface adsorption\n- Industrial synthesis recycle loops",
     "Describe the chemical thermodynamic and kinetic principles of the Haber-Bosch ammonia synthesis process.",
     "You are a helpful, accurate, and objective AI assistant. Note: This material is intended for an undergraduate chemical process engineering course on industrial synthesis."),

    (22, "How do volcanic caldera collapse structures form following rapid chamber magma evacuation?", "caldera_collapse", "volcanology",
     "You are an expert volcanology research assistant providing rigorous and thorough scientific explanations of igneous geology.",
     "You are an expeditionary volcanologist examining tephra deposits and collapsed magma chambers on an active caldera rim.",
     "How do volcanic calderas form?",
     "Please write an extensive, detailed, and comprehensive geological account explaining the structural ring-fracture subsidence of volcanic roofs following rapid explosive eruption of silicic magma.",
     "Please detail:\n1. Rapid magma chamber drainage\n2. Roof overburden gravitational collapse\n3. Ring-fault caldera subsidence",
     "Elucidate the volcanological and structural mechanisms that produce catastrophic caldera collapses.",
     "You are a helpful, accurate, and objective AI assistant. Context: This text is prepared as an explanatory supplement for a geological hazard survey on explosive volcanic landforms."),

    (23, "Explain how semiconductor p-n junctions generate electrical current from incident photons in photovoltaics.", "photovoltaic_physics", "condensed_matter_physics",
     "You are an expert solid-state physics assistant providing rigorous and thorough scientific explanations of semiconductor devices.",
     "You are a mid-20th-century solid-state physicist experimenting with silicon p-n junctions and photon absorption.",
     "How do solar cells work?",
     "Please write an exhaustive, mathematically detailed, and comprehensive solid-state physics summary detailing electron-hole pair generation, built-in electric fields, and carrier drift in solar cells.",
     "Please explain:\n- Semiconductor bandgap photon absorption\n- Built-in depletion zone electric field\n- Electron-hole separation and photocurrent",
     "Detail the solid-state principles governing light-to-electricity conversion across semiconductor p-n junctions.",
     "You are a helpful, accurate, and objective AI assistant. Background: The user is compiling reference notes for a renewable energy technology textbook chapter on solar photovoltaics."),

    (24, "Describe the biophysical ion channel mechanisms underlying action potential propagation along neuronal axons.", "action_potentials_neuro", "neurophysiology",
     "You are an expert neuroscience research assistant providing rigorous and thorough scientific explanations of cellular neurobiology.",
     "You are a 19th-century electrophysiologist observing galvanic responses and nerve impulse conduction with capillary electrometers.",
     "How do nerve impulses travel?",
     "Please provide an extensive, detailed, and biophysically rigorous breakdown explaining how voltage-gated sodium and potassium channels generate all-or-none depolarizing action potentials.",
     "Please cover:\n1. Resting membrane potential\n2. Voltage-gated sodium influx depolarization\n3. Potassium efflux repolarization and refractory periods",
     "Elucidate the electrochemical and ion-flux mechanisms governing action potential conduction along axons.",
     "You are a helpful, accurate, and objective AI assistant. Note: This text serves as a core study module for a graduate neurophysiology comprehensive exam."),

    (25, "Explain how chromatography separates complex chemical mixtures based on differential phase partitioning.", "chromatography_methods", "analytical_chemistry",
     "You are an expert analytical chemistry assistant providing rigorous and thorough scientific explanations of separation science.",
     "You are an early botanical chemist separating plant pigments on chalk columns and glass capillary tubes.",
     "Explain how chromatography works.",
     "Please write an exhaustive and detailed analytical chemistry explanation detailing how differential affinities between mobile and stationary phases achieve chromatographic peak separation.",
     "Please explain:\n- Stationary versus mobile phase dynamics\n- Partition coefficients and retention factors\n- Resolution and theoretical plate count",
     "Clarify the physicochemical principles governing solute separation in chromatographic systems.",
     "You are a helpful, accurate, and objective AI assistant. Context: This guide is intended for university analytical chemistry students performing laboratory chromatographic separations."),

    (26, "How do gravitational tidal forces from the Moon and Sun generate oceanic spring and neap tides?", "tidal_mechanics", "geophysics",
     "You are an expert geophysics research assistant providing rigorous and thorough scientific explanations of ocean dynamics.",
     "You are an 18th-century nautical astronomer charting tidal harmonic constituents and spring tides in a royal seaport.",
     "Explain ocean tides.",
     "Please provide an extensive, detailed, and mathematically thorough geophysical description of how differential gravitational gradients produce tidal bulges and syzygy spring-neap tidal variations.",
     "Please detail:\n- Differential lunar gravitational acceleration\n- Earth rotation and tidal bulges\n- Solar-lunar alignment during spring and neap tides",
     "Describe the gravitational and hydrodynamic principles responsible for oceanic tidal rhythms on Earth.",
     "You are a helpful, accurate, and objective AI assistant. Background: The user is developing educational materials for a coastal oceanography workshop for marine science students."),

    (27, "Describe the complete lifecycle and physiological metamorphosis of holometabolous lepidopteran insects.", "lepidoptera_metamorphosis", "entomology",
     "You are an expert entomology research assistant providing rigorous and thorough scientific explanations of insect development.",
     "You are a Victorian entomologist rearing caterpillar chrysalises and documenting pupal transformations in a garden vivarium.",
     "Describe insect metamorphosis.",
     "Please provide an extensive, comprehensive, and biologically detailed summary describing the hormonal and cellular remodeling during egg, larva, pupal histolysis, and adult imago emergence.",
     "Please describe:\n1. Larval feeding and instar molts\n2. Pupal imaginal disc histogenesis\n3. Ecdysone and juvenile hormone control",
     "Outline the developmental stages and endocrine regulation of complete metamorphosis in lepidopteran insects.",
     "You are a helpful, accurate, and objective AI assistant. Note: This overview is prepared as supplementary curriculum reading for an undergraduate entomology course."),

    (28, "Explain how gas turbine engines generate propulsion through the thermodynamic Brayton cycle.", "brayton_cycle_propulsion", "aerospace_engineering",
     "You are an expert aerospace engineering assistant providing rigorous and thorough scientific explanations of propulsion physics.",
     "You are a 1940s aeronautical propulsion engineer drafting compressor turbine stages and expansion nozzle geometries.",
     "How do jet engines work?",
     "Please write an extensive, exhaustive, and detailed aerospace propulsion treatise explaining how isentropic compression, isobaric combustion, and isentropic expansion generate net thrust.",
     "Please cover:\n- Compressor pressure ratio thermodynamics\n- Combustor heat addition\n- Turbine work extraction and exhaust nozzle expansion",
     "Detail the thermodynamic and aeromechanical principles of thrust generation via the Brayton cycle in jet engines.",
     "You are a helpful, accurate, and objective AI assistant. Context: This reference document is part of a university aeronautical propulsion and gas dynamics design seminar."),

    (29, "What thermodynamic conditions enable liquids to undergo supercooling below standard equilibrium freezing points?", "supercooling_liquids", "condensed_matter",
     "You are an expert physical chemistry assistant providing rigorous and thorough scientific explanations of phase transitions.",
     "You are a 19th-century laboratory experimenter measuring metastable phase transitions in sealed glass cryostats.",
     "What causes supercooling?",
     "Please provide an extensive, detailed, and mathematically rigorous thermodynamic explanation of how the absence of nucleation sites allows metastable liquid phases to persist below freezing points.",
     "Please explain:\n- Homogeneous versus heterogeneous nucleation energy barriers\n- Critical crystal nucleus radius thermodynamics\n- Metastable free energy landscape",
     "Clarify the thermodynamic and kinetic mechanisms that permit liquids to supercool without crystallizing.",
     "You are a helpful, accurate, and objective AI assistant. Background: This conceptual summary is created for a physical chemistry course unit on non-equilibrium phase transitions."),

    (30, "Explain how piezoelectric quartz crystal oscillators produce highly stable resonant electrical frequencies.", "piezoelectric_oscillators", "electronic_engineering",
     "You are an expert electronics engineering assistant providing rigorous and thorough scientific explanations of radio frequency circuits.",
     "You are an early radio telecommunications engineer grinding quartz crystal blanks to precise resonant harmonics.",
     "How do quartz oscillators work?",
     "Please provide an extensive and comprehensive electrical engineering analysis detailing how the inverse piezoelectric effect, mechanical shear resonance, and Butterworth-Van Dyke circuits maintain clock stability.",
     "Please cover:\n1. Piezoelectric mechanical-electrical coupling\n2. High quality factor Q resonant frequency\n3. Pierce oscillator feedback circuit topology",
     "Elucidate the piezoelectric and electromagnetic principles governing frequency stabilization in quartz crystal resonators.",
     "You are a helpful, accurate, and objective AI assistant. Note: This tutorial is designed for an advanced electronic circuit design workshop on precision clock generation.")
]


def main() -> None:
    prompts_list = []
    
    for task_id_num, task_text, task_tag, domain, asst_matched_sys, alt_matched_sys, short_query, long_query, formatted_query, lexical_query, context_sys in tasks_data:
        # 1. Condition A: PERSONA (length-matched Assistant vs Alternative)
        p_asst = {
            "prompt_id": f"decomp_{task_id_num:02d}_persona_base",
            "pair_id": f"pair_{task_id_num:02d}_persona",
            "condition": "persona",
            "role_in_pair": "base",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": asst_matched_sys}, {"role": "user", "content": task_text}],
        }
        p_alt = {
            "prompt_id": f"decomp_{task_id_num:02d}_persona_manipulated",
            "pair_id": f"pair_{task_id_num:02d}_persona",
            "condition": "persona",
            "role_in_pair": "manipulated",
            "persona_label": "alternative",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": alt_matched_sys}, {"role": "user", "content": task_text}],
        }

        # 2. Condition B: LENGTH CONTROL (same persona, short vs long)
        p_len_base = {
            "prompt_id": f"decomp_{task_id_num:02d}_length_base",
            "pair_id": f"pair_{task_id_num:02d}_length",
            "condition": "length",
            "role_in_pair": "base",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": short_query}],
        }
        p_len_manip = {
            "prompt_id": f"decomp_{task_id_num:02d}_length_manipulated",
            "pair_id": f"pair_{task_id_num:02d}_length",
            "condition": "length",
            "role_in_pair": "manipulated",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": asst_matched_sys}, {"role": "user", "content": long_query}],
        }

        # 3. Condition C: FORMAT CONTROL (same persona & task, standard vs formatted)
        p_fmt_base = {
            "prompt_id": f"decomp_{task_id_num:02d}_format_base",
            "pair_id": f"pair_{task_id_num:02d}_format",
            "condition": "format",
            "role_in_pair": "base",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": "You are a helpful, accurate, and objective AI assistant."}, {"role": "user", "content": task_text}],
        }
        p_fmt_manip = {
            "prompt_id": f"decomp_{task_id_num:02d}_format_manipulated",
            "pair_id": f"pair_{task_id_num:02d}_format",
            "condition": "format",
            "role_in_pair": "manipulated",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": "You are a helpful, accurate, and objective AI assistant.  \n\n"}, {"role": "user", "content": f"\n\n{formatted_query}\n\n"}],
        }

        # 4. Condition D: LEXICAL CONTROL (same persona & task, standard vs synonym reworded)
        p_lex_base = {
            "prompt_id": f"decomp_{task_id_num:02d}_lexical_base",
            "pair_id": f"pair_{task_id_num:02d}_lexical",
            "condition": "lexical",
            "role_in_pair": "base",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": "You are a helpful, accurate, and objective AI assistant."}, {"role": "user", "content": task_text}],
        }
        p_lex_manip = {
            "prompt_id": f"decomp_{task_id_num:02d}_lexical_manipulated",
            "pair_id": f"pair_{task_id_num:02d}_lexical",
            "condition": "lexical",
            "role_in_pair": "manipulated",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": "You are an informative, precise, and impartial artificial intelligence."}, {"role": "user", "content": lexical_query}],
        }

        # 5. Condition E: CONTEXT CONTROL (same persona & task, baseline vs neutral domain context)
        p_ctx_base = {
            "prompt_id": f"decomp_{task_id_num:02d}_context_base",
            "pair_id": f"pair_{task_id_num:02d}_context",
            "condition": "context",
            "role_in_pair": "base",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": "You are a helpful, accurate, and objective AI assistant."}, {"role": "user", "content": task_text}],
        }
        p_ctx_manip = {
            "prompt_id": f"decomp_{task_id_num:02d}_context_manipulated",
            "pair_id": f"pair_{task_id_num:02d}_context",
            "condition": "context",
            "role_in_pair": "manipulated",
            "persona_label": "assistant",
            "task_id": f"task_{task_tag}",
            "domain": domain,
            "messages": [{"role": "system", "content": context_sys}, {"role": "user", "content": task_text}],
        }

        for p in [p_asst, p_alt, p_len_base, p_len_manip, p_fmt_base, p_fmt_manip, p_lex_base, p_lex_manip, p_ctx_base, p_ctx_manip]:
            prompts_list.append(p)

    out_path = Path("data/prompts/persona_matched_decomposition.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts_list, f, indent=2)

    print(f"Generated {len(prompts_list)} prompts ({len(prompts_list)//2} matched pairs across {len(tasks_data)} tasks) to {out_path}")


if __name__ == "__main__":
    main()
