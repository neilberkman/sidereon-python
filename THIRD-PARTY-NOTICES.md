# Third-Party Notices

sidereon is licensed under the MIT License (see LICENSE). It contains, ports,
or reimplements algorithms from the following third-party sources. All are
permissive licenses; their required attributions are reproduced below. No
copyleft (GPL/LGPL/AGPL/MPL/EUPL/CDDL) code or dependencies are included.

--------------------------------------------------------------------------------
## RTKLIB (BSD 2-Clause)

The integer least-squares (MLAMBDA/LAMBDA) routine is a Rust port of RTKLIB's
`lambda.c`.

  Copyright (c) 2007-2020, T. Takasu, All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING IN ANY WAY
  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGE.

--------------------------------------------------------------------------------
## ERFA (BSD 3-Clause)

Nutation/precession coefficient tables and conventions are derived from ERFA
(Essential Routines for Fundamental Astronomy), itself derived from IAU SOFA.
The exact ERFA 2.0.1 license, including its SOFA heritage terms, is distributed
at `third_party_licenses/ERFA-BSD-3-Clause.txt`.

  Copyright (C) 2013-2021, NumFOCUS Foundation. All rights reserved.

--------------------------------------------------------------------------------
## SciPy (BSD 3-Clause)

The trust-region least-squares solver (`trust-region-least-squares`)
reimplements algorithms equivalent to SciPy's least-squares routines. The
exact SciPy 1.18.0 license is distributed at
`third_party_licenses/SciPy-BSD-3-Clause.txt`.

  Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
  All rights reserved.

--------------------------------------------------------------------------------
## ncompress (Unlicense)

Historical Unix-compress (`.Z`) transport decoding uses the `ncompress` Python
package. It is free and unencumbered software released into the public domain
under the Unlicense. See <https://github.com/valgur/ncompress>.

--------------------------------------------------------------------------------
## newtua-lzw-z (MIT OR Apache-2.0)

The zero-output structural validator for historical Unix `compress` (`.Z`)
archives is derived from the `newtua-lzw-z` 0.1.0 decoder core. It follows LZW
code widths, CLEAR resets, and ncompress group alignment without decoding the
payload.

This derived code is used under the MIT license option:

  MIT License

  Copyright (c) 2026 Aleksei Trankov

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

--------------------------------------------------------------------------------
## Compiled Rust dependencies (Apache-2.0 and ISC)

The Python extension's compiled Rust dependency graph includes the following
Apache-2.0 components:

- `approx` 0.5.1, Copyright 2015 Brendan Zabarauskas;
- `nalgebra` 0.33.3, Copyright 2020 Sébastien Crozet;
- `nalgebra-macros` 0.2.2, by Andreas Longva and Sébastien Crozet; and
- `simba` 0.9.1, Copyright 2020 Sébastien Crozet.

The full Apache License 2.0 text applying to those components follows.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

`libloading` 0.8.9 is included under the following ISC license:

  Copyright © 2015, Simonas Kazlauskas

  Permission to use, copy, modify, and/or distribute this software for any
  purpose with or without fee is hereby granted, provided that the above
  copyright notice and this permission notice appear in all copies.

  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
  SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
  OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
  CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

--------------------------------------------------------------------------------
## IERS Conventions Software

Sidereon's `solid_earth_tide` implementation and its private Rust helpers are a
derived work of the IERS Conventions `DEHANTTIDEINEL.F` routine and its
companions. Sidereon is independently named and is neither distributed nor
endorsed by the IERS Conventions Center. The Rust source describes the origin,
the renamed routines, and the differences from the Fortran implementation.
The complete non-test tide source distributed in the compiled extension is
included under
`third_party_source/sidereon-core-0.33.1/tides/` in wheels and source
distributions. The authoritative original is available from:

<https://iers-conventions.obspm.fr/content/chapter7/software/dehanttideinel/DEHANTTIDEINEL.F>

The original routine's exact license notice is packaged at
`third_party_licenses/IERS-CONVENTIONS-SOFTWARE-LICENSE.txt` and is also
reproduced below for readability:

  Copyright (C) 2008
  IERS Conventions Center

  ==================================
  IERS Conventions Software License
  ==================================

  NOTICE TO USER:

  BY USING THIS SOFTWARE YOU ACCEPT THE FOLLOWING TERMS AND CONDITIONS
  WHICH APPLY TO ITS USE.

  1. The Software is provided by the IERS Conventions Center ("the
     Center").

  2. Permission is granted to anyone to use the Software for any
     purpose, including commercial applications, free of charge,
     subject to the conditions and restrictions listed below.

  3. You (the user) may adapt the Software and its algorithms for your
     own purposes and you may distribute the resulting "derived work"
     to others, provided that the derived work complies with the
     following requirements:

     a) Your work shall be clearly identified so that it cannot be
        mistaken for IERS Conventions software and that it has been
        neither distributed by nor endorsed by the Center.

     b) Your work (including source code) must contain descriptions of
        how the derived work is based upon and/or differs from the
        original Software.

     c) The name(s) of all modified routine(s) that you distribute
        shall be changed.

     d) The origin of the IERS Conventions components of your derived
        work must not be misrepresented; you must not claim that you
        wrote the original Software.

     e) The source code must be included for all routine(s) that you
        distribute. This notice must be reproduced intact in any
        source distribution.

  4. In any published work produced by the user and which includes
     results achieved by using the Software, you shall acknowledge
     that the Software was used in obtaining those results.

  5. The Software is provided to the user "as is" and the Center makes
     no warranty as to its use or performance. The Center does not
     and cannot warrant the performance or results which the user may
     obtain by using the Software. The Center makes no warranties,
     express or implied, as to non-infringement of third party rights,
     merchantability, or fitness for any particular purpose. In no
     event will the Center be liable to the user for any consequential,
     incidental, or special damages, including any lost profits or lost
     savings, even if a Center representative has been advised of such
     damages, or for any claim by any third party.

  Correspondence concerning IERS Conventions software should be
  addressed as follows:

  Gerard Petit
  Internet email: gpetit[at]bipm.org
  Postal address: IERS Conventions Center
                  Time, frequency and gravimetry section, BIPM
                  Pavillon de Breteuil
                  92312 Sevres FRANCE

  or

  Brian Luzum
  Internet email: brian.luzum[at]usno.navy.mil
  Postal address: IERS Conventions Center
                  Earth Orientation Department
                  3450 Massachusetts Ave, NW
                  Washington, DC 20392

--------------------------------------------------------------------------------
## Reference algorithms (no code copied)

The following informed reimplementations from public specifications/literature;
no source code was copied:

- SGP4 / SDP4: D. Vallado et al., "Revisiting Spacetrack Report #3" (AIAA), and
  the CelesTrak reference vectors (validation only).
- Frame/time-scale conventions cross-checked against Skyfield (MIT) and the IAU
  conventions.
- Galileo NeQuick-G: reimplemented from the Galileo OS SIS ICD "Ionospheric
  Correction Algorithm for Galileo Single Frequency Users"; MODIP and CCIR data
  tables transcribed as ITU-R / EU-JRC reference data (facts).
- NRLMSISE-00: U.S. Naval Research Laboratory (public domain).
