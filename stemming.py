class PorterStemmer:
    """
    This class takes care of the stemming process for the words
    """
    consonants = "bcdfghjklmnpqrstwxz"
    special_case = "y"
    vowels = "aeiou"

    def stem(self, word):
        """
        main method where everything is performed
        """
        stem = word.strip()
        stem = self._porter_step_1(stem)
        stem = self._porter_step_2(stem)
        stem = self._porter_step_3(stem)
        stem = self._porter_step_4(stem)
        stem = self._porter_step_5(stem)
        return stem

    def _divide_into_groups(self, word):
        """
        divides word into groups
        """
        groups = []
        preceding = ""
        for index, letter in enumerate(word.lower()):
            if preceding == "":
                preceding = letter
            else:
                if self._compare_same_class(preceding, letter):
                    preceding += letter
                    if index == len(word) - 1:
                        groups.append(preceding)
                else:
                    groups.append(preceding)
                    preceding = letter
                    if index == len(word) - 1:
                        groups.append(letter)
        return groups

    def _compare_same_class(self, l1, l2):
        """
        compares to see if vowel or consonant
        """
        if l1 in self.consonants and l2 in self.consonants:
            return True
        elif l1 in self.vowels and l2 in self.vowels:
            return True
        else:
            return False

    def _determine_class(self, group):
        """
        determines if vowel or consonant
        """
        if group[0] in self.consonants:
            return 'C'
        return 'V'

    def _encode_word(self, word):
        """
        encodes word using V and C
        """
        encoded = self._divide_into_groups(word)
        classified = [self._determine_class(group) for group in encoded]
        return classified

    def _det_m(self, word):
        """
        calculates the measure m
        """
        classes = self._encode_word(word)
        if len(classes) < 2:
            return 0
        if classes[0] == 'C':
            classes = classes[1:]
        if classes[-1] == 'V':
            classes = classes[:len(classes) - 1]
        m = len(classes) // 2 if (len(classes) / 2) >= 1 else 0
        return m

    def _check_LT(self, stem, LT):
        """
        checks LT group
        """
        for letter in LT:
            if stem.endswith(letter):
                return True
        return False

    def _check_V(self, stem):
        """
        checks V group
        """
        for letter in stem:
            if letter in self.vowels:
                return True
        return False

    def _check_D(self, stem):
        """
        checks D group
        """
        if stem[-1] in self.consonants and stem[-2] in self.consonants:
            return True
        return False

    def _check_O(self, stem):
        """
        checks O group
        """
        if len(stem) < 3:
            return False
        if (stem[-3] in self.consonants) and (stem[-2] in self.vowels) and (stem[-1] in self.consonants) and (
                stem[-1] not in "wxy"):
            return True
        else:
            return False

    def _porter_step_1(self, word):
        """
        Processes plurals and past participles
        """
        stem = word
        step2B = False

        # Step1A
        if stem.endswith('sses'):
            stem = stem[:-2]
        elif stem.endswith('ies'):
            stem = stem[:-2]
        elif not stem.endswith('ss') and stem.endswith('s'):
            stem = stem[:-1]
        # Step 1B
        if len(stem) > 4:
            if stem.endswith("eed") and self._det_m(stem) > 0:
                stem = stem[:-1]
            elif stem.endswith("ed"):
                stem = stem[:-2]
                if not self._check_V(stem):
                    stem = word
                else:
                    step2B = True
        elif stem.endswith("ing"):
            stem = stem[:-3]
            if not self._check_V(stem):
                stem = word
            else:
                step2B = True

        if step2B:
            if stem.endswith("at") or stem.endswith("b1") or stem.endswith("iz"):
                stem += "e"
            elif self._check_D(stem) and not (self._check_LT(stem, "lsz")):
                stem = stem[:-1]
            elif self._det_m(stem) == 1 and self._check_O(stem):
                stem += "e"
        # Step 1C
        if self._check_V(stem) and stem.endswith('y'):
            stem = stem[:-1] + 'i'
        return stem

    def _porter_step_2(self, stem):
        """
        removes suffixes
        """
        pairs = [('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'), ('izer', 'ize'),
                      ('abli', 'able'), ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'), ('ousli', 'ous'),
                      ('ization', 'ize'),
                      ('ation', 'ate'), ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
                      ('ousness', 'ous'), ('aliti', 'al'), ('ivit', 'ive'), ('biliti', 'ble')]
        if self._det_m(stem) > 0:
            for term, subs in pairs:
                if stem.endswith(term):
                    return stem[:-len(term)] + subs
        return stem

    def _porter_step_3(self, stem):
        """
        further removing suffixes
        """
        pair_tests = [('icate', 'ic'), ('ative', ''), ('alize', 'al'), ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''),
                      ('ness', '')]
        if self._det_m(stem) > 0:
            for term, subs in pair_tests:
                if stem.endswith(term):
                    return stem[:-len(term)] + subs
        return stem

    def _porter_step_4(self, stem):
        """
        Removes more suffixes
        """
        suffixes_1 = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent']
        special_case = 'ion'
        suffixes_2 = ['ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize']
        if self._det_m(stem) > 1:
            for suffix in suffixes_1:
                if stem.endswith(suffix):
                    return stem[:-len(suffix)]
            if stem.endswith(special_case):
                temp = stem[:-len(special_case)]
                if self._check_LT(temp, 'st'):
                    return temp
            for suffix in suffixes_2:
                if stem.endswith(suffix):
                    return stem[:-len(suffix)]
        return stem

    def _porter_step_5(self, stem):
        """
        removes the final e in words with measure greater than 1
        """
        temp = stem
        # Step 5A
        if self._det_m(temp) > 1 and temp.endswith('e'):
            temp = temp[:-1]
        elif self._det_m(temp) == 1 and (not self._check_O(temp)) and temp.endswith('e') and len(temp) > 4:
            temp = temp[:-1]
        # Step 5B
        if self._det_m(temp) > 1 and self._check_D(temp) and self._check_LT(temp, 'l'):
            temp = temp[:-1]
        return temp
