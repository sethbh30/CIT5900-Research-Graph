import unittest
from code.extract_metadata_nih import is_fsrdc_related, parse_pmc_metadata

class TestFSRDCExtractor(unittest.TestCase):

    def test_is_fsrdc_related_true(self):
        text = "This study uses data from the Census Bureau and FSRDC."
        self.assertTrue(is_fsrdc_related(text))

    def test_is_fsrdc_related_false(self):
        text = "This study is about marine biology and ocean currents."
        self.assertFalse(is_fsrdc_related(text))

    def test_parse_pmc_metadata_invalid_xml(self):
        bad_xml = "<notvalid><unclosed>"
        result = parse_pmc_metadata(bad_xml)
        self.assertIsNone(result)

    def test_parse_pmc_metadata_not_fsrdc_related(self):
        xml = """<?xml version="1.0"?>
            <pmc-articleset>
                <article>
                    <front>
                        <article-meta>
                            <title-group>
                                <article-title>Marine Biology Advances</article-title>
                            </title-group>
                            <abstract><p>This study explores oceanography.</p></abstract>
                        </article-meta>
                    </front>
                </article>
            </pmc-articleset>
        """
        result = parse_pmc_metadata(xml)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
