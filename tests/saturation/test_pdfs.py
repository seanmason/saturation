from saturation.pdfs import PowerLawPDF


def test_get_power_law_pdf_round_trip():
    # Arrange
    pdf = PowerLawPDF(slope=-3, min_value=1, max_value=2)

    # Act
    p = pdf.get_value(1.5)
    result = pdf.get_inverse(p)

    # Assert
    assert result == 1.5

