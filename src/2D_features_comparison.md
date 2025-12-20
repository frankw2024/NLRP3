# 2D Features Comparison: HTS.py vs HTS_3D.py

## Summary

**HTS_3D.py does NOT fully preserve the 2D features from HTS.py**. There are significant omissions.

---

## Feature Breakdown

### HTS.py (Original) - Total: **2,230 features**

#### 1. Morgan Fingerprints
- **Size**: 2,048 bits
- **Function**: `morgan_fp(smiles, radius=2, nBits=2048)`
- **Status in HTS_3D.py**: ✅ **PRESERVED** (same: 2,048 bits)

#### 2. MACCS Keys
- **Size**: 167 bits
- **Function**: `maccs_fp(smiles)`
- **Status in HTS_3D.py**: ❌ **MISSING** (completely absent)

#### 3. Extended Physicochemical Descriptors
- **Size**: 15 descriptors
- **Function**: `extended_physchem_desc(smiles)`
- **Status in HTS_3D.py**: ⚠️ **PARTIALLY PRESERVED** (only 12 descriptors)

**HTS.py descriptors (15 total):**
1. `Descriptors.MolWt(mol)` ✅
2. `Descriptors.MolLogP(mol)` ✅
3. `Descriptors.NumRotatableBonds(mol)` ✅
4. `Descriptors.NumHAcceptors(mol)` ✅
5. `Descriptors.NumHDonors(mol)` ✅
6. `Descriptors.TPSA(mol)` ✅
7. `Descriptors.RingCount(mol)` ✅
8. `aromatic_rings` (custom count via GetSSSR) ❌ **MISSING**
9. `Descriptors.HeavyAtomCount(mol)` ✅
10. `Descriptors.NumHeteroatoms(mol)` ✅
11. `Descriptors.FractionCSP3(mol)` ✅
12. `Descriptors.NumAromaticRings(mol)` ✅
13. `Lipinski.NumHAcceptors(mol)` ❌ **MISSING** (has Descriptors version but not Lipinski)
14. `Lipinski.NumHDonors(mol)` ❌ **MISSING** (has Descriptors version but not Lipinski)
15. `QED.qed(mol)` ✅

**HTS_3D.py descriptors (12 total):**
1. `Descriptors.MolWt(mol)` ✅
2. `Descriptors.MolLogP(mol)` ✅
3. `Descriptors.NumHAcceptors(mol)` ✅
4. `Descriptors.NumHDonors(mol)` ✅
5. `Descriptors.TPSA(mol)` ✅
6. `Descriptors.NumRotatableBonds(mol)` ✅
7. `Lipinski.RingCount(mol)` ✅ (different from HTS.py's `Descriptors.RingCount`)
8. `Descriptors.NumAromaticRings(mol)` ✅
9. `Descriptors.HeavyAtomCount(mol)` ✅
10. `Lipinski.NumHeteroatoms(mol)` ✅ (different from HTS.py's `Descriptors.NumHeteroatoms`)
11. `Descriptors.FractionCSP3(mol)` ✅
12. `QED.qed(mol)` ✅

---

## Missing Features in HTS_3D.py

### 1. MACCS Keys (167 bits) - **CRITICAL MISSING**
- **Impact**: High - MACCS keys capture substructure patterns that complement Morgan fingerprints
- **Code location in HTS.py**: Lines 106-114
- **Status**: Completely absent from HTS_3D.py

### 2. Custom Aromatic Ring Count
- **Impact**: Medium - Different from `NumAromaticRings` (uses GetSSSR enumeration)
- **Code location in HTS.py**: Lines 123-132
- **Status**: Not implemented in HTS_3D.py

### 3. Lipinski.NumHAcceptors (duplicate but different)
- **Impact**: Low - HTS.py has both `Descriptors.NumHAcceptors` and `Lipinski.NumHAcceptors`
- **Status**: HTS_3D.py only has `Descriptors.NumHAcceptors`

### 4. Lipinski.NumHDonors (duplicate but different)
- **Impact**: Low - HTS.py has both `Descriptors.NumHDonors` and `Lipinski.NumHDonors`
- **Status**: HTS_3D.py only has `Descriptors.NumHDonors`

---

## Feature Count Comparison

| Component | HTS.py | HTS_3D.py | Difference |
|-----------|--------|-----------|------------|
| Morgan FP | 2,048 | 2,048 | 0 |
| MACCS Keys | 167 | 0 | **-167** |
| Physchem Desc | 15 | 12 | **-3** |
| **TOTAL** | **2,230** | **2,060** | **-170** |

---

## Specification Compliance

According to `codeGenNLRP3.txt`:
- Line 1: "Generate HTS_3D.py, based on the input file nlrp3_chembl_activities.csv, **with reference to HTS.py**"
- Line 2: "innovation is adding a 3D-aware branch to HTS Oracle"

**Interpretation**: The spec suggests HTS_3D.py should preserve the 2D features from HTS.py while adding 3D capabilities.

**Current Status**: ❌ **NOT FULLY COMPLIANT** - Missing MACCS keys and 3 physicochemical descriptors.

---

## Recommendations

To fully preserve HTS.py's 2D features, HTS_3D.py should:

1. **Add MACCS keys** (167 bits):
   ```python
   from rdkit.Chem import MACCSkeys
   
   def maccs_fp(smiles: str) -> np.ndarray:
       mol = Chem.MolFromSmiles(smiles)
       if mol is None:
           return np.zeros(167, dtype=np.float32)
       return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
   ```

2. **Add missing physicochemical descriptors**:
   - Custom aromatic ring count (via GetSSSR)
   - `Lipinski.NumHAcceptors` (in addition to `Descriptors.NumHAcceptors`)
   - `Lipinski.NumHDonors` (in addition to `Descriptors.NumHDonors`)

3. **Update `build_rdkit_feature_matrix`**:
   ```python
   def build_rdkit_feature_matrix(smiles_list: List[str]) -> np.ndarray:
       rdkit_rows = []
       for smi in tqdm(smiles_list, desc="RDKit 2D features"):
           fp = morgan_fp(smi)
           maccs = maccs_fp(smi)  # ADD THIS
           phys = physchem_features(smi)  # UPDATE to include all 15 descriptors
           rdkit_rows.append(np.concatenate([fp, maccs, phys]))  # UPDATE
       return np.vstack(rdkit_rows).astype(np.float32)
   ```

---

## Impact Assessment

- **Feature reduction**: 170 features missing (7.6% of original)
- **Information loss**: MACCS keys provide substructure patterns not captured by Morgan fingerprints
- **Model compatibility**: Models trained on HTS.py features may not work directly with HTS_3D.py features due to dimension mismatch

